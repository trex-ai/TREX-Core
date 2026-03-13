import asyncio
import datetime
import os
import signal
import time
from collections import deque
from statistics import mean
from typing import Any

import structlog
from sqlalchemy_utils import database_exists

# from TREX_Core
from TREX_Core.utils import db_utils
from TREX_Core.utils.records import Records
from TREX_Core.utils.utils import timestr_to_timestamp

logger = structlog.get_logger()

ParticipantState = dict[str, bool]


class Controller:
    """
    Sim controller takes over timing control from market.

    In RT mode, the market moves to the next round whether participants
    have finished their actions or not. In sim mode, market rounds do not
    start until all participants have joined, and auction rounds
    therefore do not advance until all participants have completed their
    actions.

    To do this, the controller needs the simulation start time and the
    participant IDs. This information comes from the config JSON file.

    1. sim start time
    2. number of participants, as well as their IDs

    The sim controller has special permission to see when participants
    join the market.
    """

    # Intialize client related data
    def __init__(self, client: Any, config: dict[str, Any]) -> None:
        self.__client = client
        self.__config = config

        # Add monitor control event
        self.__monitor_running = asyncio.Event()
        self.__monitor_running.set()  # Start in running state

        self.__learning_agents = [
            participant
            for participant in self.__config["participants"]
            if "learning" in self.__config["participants"][participant]["trader"]
            and self.__config["participants"][participant]["trader"]["learning"]
        ]

        self.__static_agents = [
            participant
            for participant in self.__config["participants"]
            if "learning" not in self.__config["participants"][participant]["trader"]
            or not self.__config["participants"][participant]["trader"]["learning"]
        ]

        self.__policy_clients = [
            participant
            for participant in self.__config["participants"]
            if self.__config["participants"][participant]["trader"]["type"]
            == "policy_client"
        ]
        self.__has_policy_clients = len(self.__policy_clients) > 0
        # TODO: do handshake if policy server is needed somewhere in monitor

        self.__participants: dict[str, ParticipantState] = {}
        self.__turn_control: dict[str, int] = {
            "total": 0,
            "online": 0,
            "ready": 0,
            "ended": 0,
        }

        self.__episodes = config["study"]["episodes"]
        self.__episode = 1
        # self.__episode = self.set_initial_episode()

        self.__start_time = config["study"]["start_time"]
        self.__time = self.__start_time

        self.__time_step_s = config["study"]["time_step_size"]
        self.__current_step = 0
        self.__end_step = self.__config["study"]["episode_steps"]
        self.__total_steps = self.__config["study"]["total_steps"]
        self.__eta_buffer: deque[float] = deque(maxlen=20)

        self.make_participant_tracker()

        self.timer_start = time.perf_counter()
        self.timer_end = 0.0
        self.market_id = self.__config["market"]["id"]
        self.status = {
            "monitor_timeout": 5,
            "registered_on_server": False,
            "market_id": self.market_id,
            "sim_started": False,
            "sim_ended": False,
            "episode_ended": False,
            "current_step": self.__current_step,
            "last_step_clock": None,
            "running_episodes": 0,
            "market_online": False,
            "market_ready": True,
            "sim_interrupted": False,
            "participants": self.__participants,
            "learning_agents": self.__learning_agents,
            "participants_online": False,
            "participants_ready": True,
            "turn_control": self.__turn_control,
            "market_turn_end": False,
        }
        if self.__has_policy_clients:
            # self.status['policy_sever_online'] = False
            self.status["policy_server_ready"] = False
        # self.training_controller = TrainingController(self.__config, self.status)

        if "records" in config:
            output_db_str = db_utils.make_db_str(
                db_utils.get_credentials(),
                config["database"],
                config["database"]["output_db"],
            )
            self.records = Records(db_string=output_db_str, columns=config["records"])

        self._write_state_lock = asyncio.Lock()
        self.__report_steps = self.__config["study"].get(
            "report_steps", self.__end_step
        )

    async def delay(self, seconds: float) -> None:
        """Delay the simulation without interrupting thread control.

        Params:
            int or float: Number of seconds to delay.
        """
        await asyncio.sleep(seconds)

    def get_start_time(self) -> int:
        tz_str = self.__config["study"]["timezone"]
        dt_str = self.__config["study"]["start_datetime"]
        return timestr_to_timestamp(dt_str, tz_str)

    # Initialize data for participant turns
    def make_participant_tracker(self) -> None:
        self.__turn_control["total"] = len(self.__config["participants"])
        for participant_id in list(self.__config["participants"]):
            self.__participants[participant_id] = {
                "online": False,
                "turn_end": False,
                "ready": False,
            }

    # Set intial generation data folder
    # TODO: update this to check all dbs
    def set_initial_episode(self):
        db_string = self.__config["study"]["output_database"]
        if not database_exists(db_string):
            return 0

        # TODO: rewrite generation detection for resume
        # if self.__config['study']['resume']:
        #     pass
        # return 0

        return 0

    # Register client in server
    async def register(self):
        self.status["registered_on_server"] = True

    # Track ending of turns
    async def update_turn_status(self, participant_id):
        async with self._write_state_lock:
            if participant_id in self.__participants:
                self.__participants[participant_id]["turn_end"] = True
                self.__turn_control["ended"] += 1

            if self.__turn_control["ended"] < self.__turn_control["total"]:
                return

            if self.__has_policy_clients and not self.status["policy_server_ready"]:
                return

            if self.status["market_turn_end"]:
                await self.__advance_turn()

    def __reset_turn_trackers(self):
        self.status["market_turn_end"] = False
        for participant_id in self.__participants:
            self.__participants[participant_id]["turn_end"] = False
        self.__turn_control["ended"] = 0

        if self.__has_policy_clients:
            self.status["policy_server_ready"] = False

    async def __advance_turn(self):
        # Once all participants have gone through their turn,
        # and market is ready
        # reset turn trackers
        # take next step
        self.__reset_turn_trackers()
        self.__time += self.__time_step_s
        await self.step()

    # Reset turn count
    async def market_turn_end(self):
        self.status["market_turn_end"] = True

    # Update tracker when participant is active
    async def participant_status(self, participant_id, status, condition):
        async with self._write_state_lock:
            if participant_id in self.__participants:
                last_condition = self.__participants[participant_id][status]
                if condition == last_condition:
                    return
                self.__participants[participant_id][status] = condition
                if condition:
                    self.__turn_control[status] = min(
                        self.__turn_control["total"], self.__turn_control[status] + 1
                    )
                else:
                    self.__turn_control[status] = max(
                        0, self.__turn_control[status] - 1
                    )
            if self.__turn_control[status] < self.__turn_control["total"]:
                self.status["participants_" + status] = False
            else:
                self.status["participants_" + status] = True

    # Update tracker when participant is active
    async def participant_online(self, participant_id, online):
        async with self._write_state_lock:
            if not self.__participants.get(participant_id):
                return
            if not self.__participants[participant_id]["online"] ^ online:
                return
            self.__participants[participant_id]["online"] = online
            if online:
                self.__turn_control["online"] = min(
                    self.__turn_control["total"], self.__turn_control["online"] + 1
                )
            else:
                self.__turn_control["online"] = max(
                    0, self.__turn_control["online"] - 1
                )
                self.status["sim_interrupted"] = True
                self.status["sim_started"] = False
            if self.__turn_control["online"] < self.__turn_control["total"]:
                self.status["participants_online"] = False
            else:
                self.status["participants_online"] = True

    async def __handle_monitor_blockers(self) -> bool:
        blocked = True

        if not self.status["market_online"]:
            self.__client.publish(
                f"{self.market_id}/simulation/is_market_online", "", qos=1
            )
        elif not self.status["participants_online"]:
            self.__client.publish(
                f"{self.market_id}/simulation/is_participant_joined",
                "",
                qos=1,
                user_property=[("to", "^all")],
            )
        elif not self.status["market_ready"]:
            blocked = True
        elif self.__has_policy_clients and not self.status["policy_server_ready"]:
            self.__client.publish(
                f"{self.market_id}/simulation/is_policy_server_online", "", qos=1
            )
        else:
            blocked = self.status["sim_ended"] or not self.status["participants_ready"]
            if self.status["sim_interrupted"]:
                # TODO: Recover the turn tracker if a participant disconnects
                # mid-turn so the simulation can resume from a safe state.
                logger.warning("sim interrupted")
                if self.__turn_control["total"] - self.__turn_control["online"] > 1:
                    self.__current_step = 0
                else:
                    await self.__client.emit("re_register_participant")
                self.status["sim_interrupted"] = False
                blocked = True

        return blocked

    async def monitor(self) -> None:
        first_cycle = True
        while True:
            if not self.__monitor_running.is_set():
                await self.__monitor_running.wait()

            if first_cycle:
                first_cycle = False
            else:
                await self.delay(self.status["monitor_timeout"])

            if not self.status["registered_on_server"] or self.status["sim_started"]:
                continue

            if await self.__handle_monitor_blockers():
                continue

            self.status["sim_started"] = True

            await self.pause_monitor()
            await self.__advance_turn()

    async def __print_step_time(self, report_steps: int | None = None) -> None:
        if report_steps is None:
            report_steps = self.__end_step
        if self.__current_step % report_steps == 0 and self.__current_step:
            # Print time information for time step/ expected runtime
            end = time.perf_counter()
            step_time = end - self.timer_start
            self.__eta_buffer.append(step_time)
            elapsed_steps = self.__current_step + (self.__episode - 1) * self.__end_step
            steps_to_go = self.__total_steps - elapsed_steps
            eta_s = steps_to_go * mean(self.__eta_buffer) / report_steps
            # replace with structured log
            logger.info(
                "Step time report",
                market_id=self.__config["market"]["id"],
                episode=self.__episode,
                current_step=self.__current_step,
                step_time=round(step_time, 1),
                eta=str(datetime.timedelta(seconds=eta_s)),
            )
            self.timer_start = time.perf_counter()

    async def step(self):
        self.status["last_step_clock"] = time.time()

        if not self.status["sim_started"]:
            return

        # Beginning new episode
        if self.__current_step == 0:
            # use structlog
            logger.info(
                "Starting simulation episode",
                episode=self.__episode,
                market_id=self.market_id,
            )
            if hasattr(self, "records"):
                table_name = f"{self.__episode}_{self.market_id}"
                await self.records.create_table(table_name)

            self.__client.publish(
                f"{self.market_id}/simulation/start_episode",
                self.__episode,
                user_property=[("to", "^all")],
                qos=1,
            )
            self.status["episode_ended"] = False

        # Beginning new time step
        if self.__current_step <= self.__end_step:
            await self.__print_step_time(self.__report_steps)
            self.__current_step += 1

            message = {
                "time": self.__time,
                "duration": self.__time_step_s,
                "update": True,
            }
            self.__client.publish(
                f"{self.market_id}/simulation/start_round",
                message,
                user_property=[("to", "^all")],
                qos=1,
            )
        # end of episode
        elif self.__current_step == self.__end_step + 1:
            self.__turn_control.update(
                {
                    "ready": 0
                    # 'weights_loaded': 0,
                    # 'weights_saved': 0
                }
            )
            for participant_id in self.__participants:
                self.__participants[participant_id].update(
                    {
                        "ready": False
                        # 'weights_loaded': False,
                        # 'weights_saved': False
                    }
                )
            self.status["participants_ready"] = False

            self.status["episode_ended"] = True
            # end simulation if the final generation is done, else reset step and stuff
            if self.__episode < self.__episodes:
                logger.info(
                    "Episode complete", episode=self.__episode, market_id=self.market_id
                )
                self.__episode += 1
                self.status["running_episodes"] += 1
                self.__current_step = 0
                self.__start_time = self.get_start_time()
                self.__time = self.__start_time
                self.status["sim_started"] = False
                self.status["market_ready"] = False

                message = {
                    # 'output_path': self.status['output_path'],
                    # 'db_path': self.__config['study']['output_database'],
                    "episode": self.__episode - 1,
                    "market_id": self.__config["market"]["id"],
                }
                self.__client.publish(
                    f"{self.market_id}/simulation/end_episode",
                    message,
                    user_property=[("to", "^all")],
                    qos=1,
                )
                await self.resume_monitor()
            else:
                logger.info(
                    "Episode complete", episode=self.__episode, market_id=self.market_id
                )
                self.status["sim_ended"] = True
                # TODO: add function to reset sim for next hyperparameter set
                # if self.status['sim_ended']:
                logger.info(
                    "End simulation",
                    episode=self.__episode,
                    total_episodes=self.__episodes,
                    market_id=self.market_id,
                )
                self.__client.publish(
                    f"{self.market_id}/simulation/end_simulation",
                    self.market_id,
                    user_property=[("to", "^all")],
                    qos=1,
                )
                await self.delay(1)
                self.__client._trex_shutdown_reason = "simulation_complete"
                logger.info(
                    "Sim controller disconnecting after simulation completion",
                    market_id=self.market_id,
                    episode=self.__episode,
                )
                await self.__client.disconnect()
                os.kill(os.getpid(), signal.SIGINT)

    async def pause_monitor(self) -> None:
        """Pause the monitor loop without cancelling the task"""
        if self.__monitor_running.is_set():
            logger.info("Pausing monitor - simulation active", market_id=self.market_id)
            self.__monitor_running.clear()

    async def resume_monitor(self) -> None:
        """Resume the monitor loop"""
        if not self.__monitor_running.is_set():
            logger.info(
                "Resuming monitor - simulation paused/between episodes",
                market_id=self.market_id,
            )
            self.__monitor_running.set()

    @property
    def current_step(self) -> int:
        """Read-only property."""
        return self.__current_step
