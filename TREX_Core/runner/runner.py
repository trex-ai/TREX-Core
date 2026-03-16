import json
import os
import subprocess  # nosec B404
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from importlib import import_module
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import commentjson
import numpy as np
import sqlalchemy
import structlog
from packaging import version
from sqlalchemy import Column, MetaData, create_engine, insert, select
from sqlalchemy.orm import Session
from sqlalchemy_utils import create_database, database_exists, drop_database

from TREX_Core.runner.make import participant, sim_controller
from TREX_Core.utils import db_utils, utils

logger = structlog.get_logger()

ConfigDict = dict[str, Any]
LaunchCommand = tuple[str, list[str]]
OptionalLaunchCommand = tuple[str | None, list[str] | None]


def _iter_config_roots(root_dir: str = "") -> Iterator[Path]:
    roots: list[Path] = []
    if root_dir:
        supplied_root = Path(root_dir).expanduser().resolve()
        roots.extend((supplied_root, supplied_root / "TREX_Core"))
    else:
        env_root = os.environ.get("TREX_CORE_ROOT", "").strip()
        if env_root:
            env_root_path = Path(env_root).expanduser().resolve()
            roots.extend((env_root_path, env_root_path / "TREX_Core"))

        cwd = Path.cwd().resolve()
        roots.extend((cwd, cwd / "TREX_Core"))
        roots.append(Path(__file__).resolve().parents[1])

    seen: set[str] = set()
    for root in roots:
        root_str = str(root)
        if root_str in seen:
            continue
        seen.add(root_str)
        yield root


def _resolve_config_file(config_name: str, root_dir: str = "") -> tuple[str, str]:
    searched: list[str] = []
    for root in _iter_config_roots(root_dir):
        config_file = root / "configs" / f"{config_name}.json"
        searched.append(str(config_file))
        if config_file.is_file():
            return str(config_file), str(root)

    searched_paths = "\n".join(searched)
    raise FileNotFoundError(
        f'Unable to locate config "{config_name}.json". Searched:\n{searched_paths}'
    )


def get_config(
    config_name: str,
    original: bool = False,
    root_dir: str = "",
    **kwargs: Any,
) -> ConfigDict:
    if not root_dir:
        root_dir = str(kwargs.get("root_dir", ""))
    config_file, resolved_root_dir = _resolve_config_file(config_name, root_dir)

    config = _load_json_file(config_file)
    if not isinstance(config, dict):
        raise TypeError(f"Config {config_name!r} must deserialize to a dictionary")

    if original:
        return config
    config["study"]["root_dir"] = resolved_root_dir
    config["study"]["checkpoint_save_path"] = os.path.join(
        resolved_root_dir, "checkpoint"
    )

    if "name" in config["study"] and config["study"]["name"]:
        study_name = config["study"]["name"].replace(" ", "_")
    else:
        study_name = config_name

    config["study"]["name"] = study_name
    config["database"]["output_db"] = study_name

    return config


def _load_json_file(file_path: str | Path) -> Any:
    with open(file_path, encoding="utf-8") as file_handle:
        return commentjson.load(file_handle)


class Runner:
    def __init__(self, config: str, resume: bool = False, **kwargs: Any) -> None:
        self.resume = resume
        self.purge_db = bool(kwargs.get("purge", False))
        self.config_file_name = config
        self.config_original = get_config(config, original=True)
        self.config = get_config(config)
        self.__config_version_valid = bool(
            version.parse(self.config["version"]) >= version.parse("5.1.0")
        )

    # Give starting time for simulation
    def __get_start_time(self) -> int:
        start_datetime = self.config["study"]["start_datetime"]
        start_timezone = self.config["study"]["timezone"]
        if not isinstance(start_datetime, str):
            raise TypeError("study.start_datetime must be a string")
        return utils.timestr_to_timestamp(start_datetime, str(start_timezone))

    def __create_sim_metadata(self, config):
        db_string = config["study"]["output_database"]

        engine = create_engine(db_string)
        if not sqlalchemy.inspect(engine).has_table("metadata"):
            self.__create_metadata_table(db_string)

        table = db_utils.get_table(db_string, "metadata", engine)
        data = []

        for _episode in range(config["study"]["episodes"]):
            start_time = self.__get_start_time()
            data.append(
                {
                    "start_timestamp": start_time,
                    "end_timestamp": int(
                        start_time + self.config["study"]["days"] * 1440
                    ),
                }
            )

        with Session(engine) as session:
            session.execute(insert(table), data)
            session.commit()

    def __create_sim_db(self, db_string, config):
        if not database_exists(db_string):
            engine = create_engine(db_string)
            db_utils.create_db(db_string=db_string, engine=engine)
            self.__create_configs_table(db_string)

            table = db_utils.get_table(db_string, "configs", engine)
            with Session(engine) as session:
                session.execute(insert(table), ({"id": 0, "data": config}))
                session.commit()

    def __create_table(self, db_string, table):
        engine = create_engine(db_string)
        if not database_exists(engine.url):
            create_database(engine.url)
        table.create(engine, checkfirst=True)

    def __create_configs_table(self, db_string):
        table = sqlalchemy.Table(
            "configs",
            MetaData(),
            Column("id", sqlalchemy.Integer, primary_key=True),
            Column("data", sqlalchemy.JSON),
        )
        self.__create_table(db_string, table)

    def __create_metadata_table(self, db_string):
        table = sqlalchemy.Table(
            "metadata",
            MetaData(),
            Column("episode", sqlalchemy.Integer, primary_key=True),
            Column("data", sqlalchemy.JSON),
        )
        self.__create_table(db_string, table)

    def __ensure_server_defaults(self, config: ConfigDict) -> None:
        server_config = config.get("server")
        if not isinstance(server_config, dict):
            server_config = {}
            config["server"] = server_config
        if not server_config.get("host"):
            server_config["host"] = "localhost"
        if not server_config.get("port"):
            server_config["port"] = 42069

    def __merge_default_participant_configs(self, config: ConfigDict) -> None:
        participants = config["participants"]
        default_participant_configs = participants.pop("_default", {})
        for participant_id, participant_config in participants.items():
            participants[participant_id] = (
                default_participant_configs | participant_config
            )

    def __get_learning_participants(self, config: ConfigDict) -> list[str]:
        return [
            participant_id
            for participant_id, participant_config in config["participants"].items()
            if participant_config["trader"].get("learning")
        ]

    def __get_policy_server_names(self, config: ConfigDict) -> list[str]:
        return [key for key in config if key.endswith("_policy_server")]

    def __drop_policy_servers(
        self, config: ConfigDict, policy_servers: Sequence[str]
    ) -> None:
        for server_name in policy_servers:
            config.pop(server_name, None)

    def __configure_baseline(
        self,
        config: ConfigDict,
        simulation_type: str,
        policy_servers: Sequence[str],
    ) -> None:
        config["study"]["episodes"] = 1
        config["market"]["id"] = simulation_type
        config["market"]["save_transactions"] = True
        for participant_config in config["participants"].values():
            trader = participant_config["trader"]
            trader.update({"learning": False, "type": "baseline_agent"})
            if "actions" in trader:
                trader["actions"].pop("replay", None)
        self.__drop_policy_servers(config, policy_servers)

    def __configure_training(
        self,
        config: ConfigDict,
        simulation_type: str,
        learning_participants: Sequence[str],
        has_policy_clients: bool,
        policy_servers: Sequence[str],
    ) -> None:
        config["market"]["id"] = simulation_type
        config["market"]["save_transactions"] = True
        for participant_id in learning_participants:
            trader = config["participants"][participant_id]["trader"]
            trader["learning"] = True
            trader["study_name"] = config["study"]["name"]
        if not has_policy_clients:
            self.__drop_policy_servers(config, policy_servers)

    def __configure_replay(self, config: ConfigDict, simulation_type: str) -> None:
        config["market"]["id"] = simulation_type
        config["market"]["save_transactions"] = False
        for participant_config in config["participants"].values():
            participant_config["trader"]["learning"] = False

    def __get_energy_profile_names(self, config: ConfigDict) -> set[str]:
        energy_profile_names: set[str] = set()
        for participant_id, participant_config in config["participants"].items():
            trader = participant_config["trader"]
            profile_name = trader.get("use_synthetic_profile", participant_id)
            energy_profile_names.add(str(profile_name))
        return energy_profile_names

    def __get_profile_time_step_size(self, config: ConfigDict, start_time: int) -> int:
        energy_profile_names = self.__get_energy_profile_names(config)
        random_check = utils.secure_random.sample(
            list(energy_profile_names), min(len(energy_profile_names), 5)
        )
        interval_checks: list[int] = []

        root_dir = config["study"]["root_dir"]
        profile_db_str = db_utils.make_db_str(
            db_utils.get_credentials(root_dir),
            self.config["database"],
            self.config["database"]["profiles_db"],
        )

        engine = create_engine(profile_db_str)
        with Session(engine) as session:
            for profile_name in random_check:
                table = db_utils.get_table(profile_db_str, profile_name, engine)
                stm = select(table.c.time).where(table.c.time >= start_time).fetch(100)
                out = session.execute(stm).all()
                out_array = np.array(out)
                unique_intervals = np.unique((out_array - np.roll(out_array, 1))[1:])
                if unique_intervals.size > 1:
                    raise ValueError(
                        f"Profile {profile_name} time intervals are not consistent"
                    )
                interval_checks.append(int(unique_intervals[0]))

        profile_set_interval_check = np.unique(interval_checks)
        if profile_set_interval_check.size > 1:
            raise ValueError("Profile set time intervals are not consistent")
        return int(profile_set_interval_check[0])

    def __update_study_timing(
        self, config: ConfigDict, start_time: int, time_step_size: int
    ) -> None:
        config["study"]["time_step_size"] = time_step_size
        day_steps = int(1440 / (config["study"]["time_step_size"] / 60))
        episodes = config["study"]["episodes"]
        episode_steps = int(config["study"]["days"] * day_steps)
        total_steps = episodes * episode_steps
        end_time = start_time + episode_steps
        logger.debug(
            "Config modified",
            start_time=start_time,
            end_time=end_time,
            episodes=episodes,
            episode_steps=episode_steps,
            total_steps=total_steps,
        )
        config["study"].update(
            {
                "start_time": start_time,
                "end_time": end_time,
                "episodes": episodes,
                "episode_steps": episode_steps,
                "total_steps": total_steps,
            }
        )

    def modify_config(self, simulation_type: str, **_kwargs: Any) -> ConfigDict:
        config: ConfigDict = json.loads(json.dumps(self.config))
        self.__ensure_server_defaults(config)
        config["study"]["type"] = simulation_type
        self.__merge_default_participant_configs(config)

        learning_participants = self.__get_learning_participants(config)
        has_policy_clients = any(
            participant_config["trader"].get("type") == "policy_client"
            for participant_config in config["participants"].values()
        )
        policy_servers = self.__get_policy_server_names(config)

        if simulation_type == "baseline":
            self.__configure_baseline(config, simulation_type, policy_servers)
        elif simulation_type == "training":
            self.__configure_training(
                config,
                simulation_type,
                learning_participants,
                has_policy_clients,
                policy_servers,
            )
        elif simulation_type == "replay":
            self.__configure_replay(config, simulation_type)

        start_datetime = config["study"]["start_datetime"]
        timezone = config["study"]["timezone"]
        start_time = utils.timestr_to_timestamp(start_datetime, timezone)
        time_step_size = self.__get_profile_time_step_size(config, start_time)
        self.__update_study_timing(config, start_time, time_step_size)
        return config

    def __append_launch_command(
        self,
        launch_list: list[LaunchCommand],
        command: OptionalLaunchCommand,
    ) -> None:
        target, target_args = command
        if target is None or target_args is None:
            return
        launch_list.append((target, target_args))

    def make_launch_list(
        self,
        config: ConfigDict | None = None,
        skip: str | tuple[str, ...] = (),
    ) -> list[LaunchCommand]:
        if config is None:
            config = self.config

        parallel = config["study"].get("parallel", 1)
        launch_list: list[LaunchCommand] = []
        base_market_id = config["market"].get("id") or config["market"]["type"]
        skip_items = (skip,) if isinstance(skip, str) else skip

        for idx in range(parallel):
            config["market"]["id"] = (
                f"{base_market_id}/{idx}" if parallel > 1 else base_market_id
            )

            exclude = {
                "version",
                "study",
                "server",
                "database",
                "records",
                "participants",
                "_default",
            }
            exclude.update(skip_items)

            dynamic = [k for k in config if k not in exclude]
            for module_n in dynamic:
                if module_n in exclude:
                    continue

                p_copies = config[module_n].get("parallel_copies")
                if p_copies and p_copies > idx + 1:
                    continue

                try:
                    module = import_module("TREX_Core.runner.make." + module_n)
                    self.__append_launch_command(launch_list, module.cli(config))
                except ImportError:
                    logger.warning("Module not found for launch", module=module_n)
                    module = import_module("runner.make." + module_n)
                    self.__append_launch_command(launch_list, module.cli(config))
            if "sim_controller" not in exclude:
                self.__append_launch_command(launch_list, sim_controller.cli(config))

            for p_id in config["participants"]:
                if p_id not in exclude:
                    self.__append_launch_command(
                        launch_list, participant.cli(config, p_id)
                    )

        logger.info("Launch list created", launch_count=len(launch_list))
        # logger.debug("Launch list", launch_list=launch_list)
        return launch_list

    def run_subprocess(
        self, args: LaunchCommand, delay: float = 0, **kwargs: Any
    ) -> None:
        time.sleep(delay)
        target, target_args = args
        command = [sys.executable]
        if os.path.sep in target or target.endswith(".py"):
            command.append(target)
        else:
            command.extend(["-m", target])
        check = kwargs.pop("check", False)
        subprocess.run(  # noqa: S603  # nosec B603
            [*command, *target_args],
            check=check,
            **kwargs,
        )

    def __get_pool_size(self, requested_size: int | None, launch_count: int) -> int:
        pool_size = requested_size if requested_size is not None else launch_count
        return max(pool_size, cpu_count() - 5)

    def run(self, launch_list: Sequence[LaunchCommand], **kwargs: Any) -> None:
        if not self.__config_version_valid:
            logger.error("CONFIG NOT COMPATIBLE")
            return
        if len(launch_list) == 1:
            logger.info("Running single launch", launch=launch_list[0])
            self.run_subprocess(launch_list[0])
        else:
            pool_size = self.__get_pool_size(kwargs.get("pool_size"), len(launch_list))
            pool = Pool(pool_size)
            pool.map(self.run_subprocess, launch_list)
            pool.close()

    def run_simulations(self, simulations: Iterable[str], **kwargs: Any) -> None:
        if not self.__config_version_valid:
            logger.error("CONFIG NOT COMPATIBLE")
            return

        root_dir = self.config["study"]["root_dir"]
        credentials = db_utils.get_credentials(root_dir)
        database_config = self.config["database"]
        db_string = db_utils.make_db_str(
            credentials, database_config, self.config["study"]["name"]
        )

        if self.purge_db and database_exists(db_string):
            drop_database(db_string)
        self.__create_sim_db(db_string, self.config_original)

        simulations_list = []
        launch_list: list[LaunchCommand] = []

        for simulation in simulations:
            simulations_list.append({"simulation_type": simulation})

        for sim_param in simulations_list:
            config = self.modify_config(**sim_param)
            launch_list.extend(self.make_launch_list(config, **kwargs))
        pool_size = self.__get_pool_size(kwargs.get("pool_size"), len(launch_list))
        pool = Pool(pool_size)
        pool.map(self.run_subprocess, launch_list)
        pool.close()
