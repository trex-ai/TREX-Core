import ast
import asyncio
import importlib
import os
import signal
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypedDict, cast

import databases
from async_lru import alru_cache
from cuid2 import Cuid
from gmqtt import Client as MQTTClient

from TREX_Core.participants import ledger
from TREX_Core.utils import db_utils, utils
from TREX_Core.utils.records import Records

type TimeInterval = ledger.TimeInterval
type ProfileValues = tuple[float, float]
type ActionMap = dict[str, Any]


class MeterGeneration(TypedDict):
    solar: float
    bess: float


class MeterLoadBess(TypedDict):
    solar: float


class MeterLoadOther(TypedDict):
    solar: float
    bess: float
    ext: float


class MeterLoad(TypedDict):
    bess: MeterLoadBess
    other: MeterLoadOther


class MeterData(TypedDict):
    generation: MeterGeneration
    load: MeterLoad


@dataclass
class StorageContext:
    get_info: Callable[..., dict[str, Any]]
    check_schedule: Callable[[TimeInterval], Awaitable[dict[TimeInterval, Any]]]


@dataclass
class TraderContext:
    client: MQTTClient
    participant_id: str
    market_id: str
    timing: dict[str, Any]
    ledger: ledger.Ledger
    extra_tx: dict[str, Any]
    market_info: dict[Any, Any]
    metadata: dict[str, Any] | None
    read_profile: Callable[[TimeInterval], Awaitable[ProfileValues]]
    meter: Callable[[], MeterData]
    storage: StorageContext | None = None


@dataclass
class RecordsContext:
    participant_id: str
    timing: dict[str, Any]
    next_actions: Callable[[], ActionMap] | None
    meter: Callable[[], dict[str, Any]] | None
    read_profile: Callable[[TimeInterval], Awaitable[ProfileValues]] | None
    storage: Any | None = None
    trader: Any | None = None


class Participant:
    """
    Participant is the interface layer between local resources and the Market
    """

    storage: Any | None
    records: Records | None
    action_replay: dict[str, Any] | None

    def __init__(
        self,
        client: MQTTClient,
        participant_id: str,
        market_id: str,
        database_config: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        self.__initialize_state(
            client=client,
            participant_id=participant_id,
            market_id=market_id,
            database_config=database_config,
            sid=cast(str, kwargs.get("sid", market_id)),
        )

        trader_params_raw = kwargs.get("trader")
        if not isinstance(trader_params_raw, dict):
            raise TypeError("Trader configuration must be a dictionary.")
        trader_params = dict(trader_params_raw)

        storage_params: dict[str, Any] = {}
        storage_params_raw = kwargs.get("storage")
        if storage_params_raw is not None:
            if not isinstance(storage_params_raw, dict):
                raise TypeError("Storage configuration must be a dictionary.")
            storage_params = dict(storage_params_raw)

        # Initialize trader variables and functions
        storage_ctx: StorageContext | None = None
        if storage_params:
            storage_type = storage_params.pop("type", None)
            if not isinstance(storage_type, str) or not storage_type:
                raise ValueError("Storage configuration must include a non-empty type.")
            self.storage = importlib.import_module(
                f"TREX_Core.devices.{storage_type}"
            ).Storage(**storage_params)
            self.storage.timing = self.__timing
            storage_ctx = StorageContext(
                get_info=self.storage.get_info,
                check_schedule=self.storage.check_schedule,
            )

        trader_ctx = TraderContext(
            client=self.__client,
            participant_id=self.participant_id,
            market_id=self.market_id,
            timing=self.__timing,
            ledger=self.__ledger,
            extra_tx=self.__extra_transactions,
            market_info=self.__market_info,
            metadata=self.__trader_metadata,
            read_profile=self.__read_profile,
            meter=lambda: self.__meter,
            storage=storage_ctx,
        )

        actions_config = trader_params.get("actions")
        if isinstance(actions_config, dict):
            replay_config = actions_config.get("replay")
            if isinstance(replay_config, dict):
                self.action_replay = dict(replay_config)

        self.__generation_scale = float(kwargs.get("generation", {}).get("scale", 1))
        self.__load_scale = float(kwargs.get("load", {}).get("scale", 1))
        synthetic_profile = trader_params.pop("use_synthetic_profile", None)
        self.__synthetic_profile = (
            synthetic_profile if isinstance(synthetic_profile, str) else None
        )

        trader_type = trader_params.pop("type", None)
        if not isinstance(trader_type, str) or not trader_type:
            raise ValueError("Trader configuration must include a non-empty type.")

        try:
            trader_module = importlib.import_module(f"traders.{trader_type}")
        except ImportError:
            trader_module = importlib.import_module(f"TREX_Core.traders.{trader_type}")
        self.trader = trader_module.Trader(context=trader_ctx, **trader_params)

        records_ctx = RecordsContext(
            participant_id=self.participant_id,
            timing=self.__timing,
            next_actions=lambda: self.__next_actions,
            read_profile=self.__read_profile,
            meter=lambda: cast(dict[str, Any], self.__meter),
            storage=storage_ctx,
            trader=trader_ctx,
        )
        if "records" in kwargs:
            output_db_str = db_utils.make_db_str(
                db_utils.get_credentials(),
                self.__database_config,
                self.__database_config["output_db"],
            )
            self.records = Records(
                db_string=output_db_str, columns=kwargs["records"], context=records_ctx
            )

    @staticmethod
    def __new_meter() -> MeterData:
        return {
            "generation": {"solar": 0.0, "bess": 0.0},
            "load": {
                "bess": {"solar": 0.0},
                "other": {"solar": 0.0, "bess": 0.0, "ext": 0.0},
            },
        }

    def __initialize_state(
        self,
        *,
        client: MQTTClient,
        participant_id: str,
        market_id: str,
        database_config: dict[str, Any],
        sid: str,
    ) -> None:
        self.server_online = False
        self.busy = False
        self.run = True
        self.market_id = market_id
        self.market_connected = False
        self.participant_id = str(participant_id)
        self.sid = sid
        self.market_sid = market_id
        self.timezone = "UTC"
        self.__client = client
        self.__database_config = database_config
        self.__profile: dict[str, Any] = {}
        self.__ledger = ledger.Ledger(self.participant_id)
        self.__extra_transactions: dict[str, Any] = {}
        self.__market_info: dict[Any, Any] = {}
        self.__trader_metadata: dict[str, Any] = {}
        self.__meter = self.__new_meter()
        self.__timing: dict[str, Any] = {}
        self.__next_actions: ActionMap = {}
        self.storage = None
        self.records = None
        self.action_replay = None

    async def open_db(self) -> None:
        """Open connections for profile data and optional replay records."""
        credentials = db_utils.get_credentials()
        profile_db_str = db_utils.make_db_str(
            credentials, self.__database_config, self.__database_config["profiles_db"]
        )
        await self.open_profile_db(profile_db_str)

        if self.action_replay is not None:
            records_table = (
                f"{self.action_replay['episode']}_"
                f"{self.action_replay['market']}_records"
            )
            records_db_str = db_utils.make_db_str(
                credentials, self.__database_config, self.action_replay["study"]
            )
            await self.open_action_replay_db(records_db_str, records_table)

    async def open_profile_db(self, db_path: str) -> None:
        self.__profile["db"] = databases.Database(db_path)
        profile_name = self.__synthetic_profile or self.participant_id
        self.__profile["name"] = profile_name
        self.__profile["db_table"] = db_utils.get_table(db_path, profile_name)
        if self.__profile["db_table"] is not None:
            await self.__profile["db"].connect()

    async def open_action_replay_db(self, db_path: str, table_name: str) -> None:
        if self.action_replay is None:
            return
        self.action_replay["db"] = databases.Database(db_path)
        self.action_replay["db_table"] = db_utils.get_table(db_path, table_name)
        if self.action_replay["db_table"] is not None:
            await self.action_replay["db"].connect()

    async def join_market(self) -> bool:
        """Emits event to join a Market"""
        if self.market_connected:
            return True

        client_data = {
            "type": ("participant", "Residential"),
            "id": self.participant_id,
            "sid": self.sid,
            "market_id": self.market_id,
        }
        self.__client.publish(
            f"{self.market_id}/join_market/{self.participant_id}",
            client_data,
            retain=True,
            qos=1,
            user_property=[("to", "^all")],
        )
        return False

    async def update_extra_transactions(self, message: dict[str, Any]) -> None:
        time_delivery = tuple(message.pop("time_delivery"))
        # TODO: recreate the simplified extra transactions here

        grid_transactions = message["grid"]
        for idx in range(len(grid_transactions["sell"])):
            transaction = grid_transactions["sell"][idx]
            transaction_record = {
                "quantity": transaction[0],
                "seller_id": self.participant_id,
                "buyer_id": "grid",
                "energy_source": transaction[2],
                "settlement_price_sell": transaction[1],
                "settlement_price_buy": transaction[1],
                "time_creation": time_delivery[0],
                "time_purchase": time_delivery[1],
                "time_consumption": time_delivery[1],
            }
            grid_transactions["sell"][idx] = transaction_record.copy()

        for idx in range(len(grid_transactions["buy"])):
            transaction = grid_transactions["buy"][idx]
            transaction_record = {
                "quantity": transaction[0],
                "seller_id": "grid",
                "buyer_id": self.participant_id,
                "energy_source": "grid",
                "settlement_price_sell": transaction[1],
                "settlement_price_buy": transaction[1],
                "time_creation": time_delivery[0],
                "time_purchase": time_delivery[1],
                "time_consumption": time_delivery[1],
            }
            grid_transactions["buy"][idx] = transaction_record.copy()

        self.__ledger.extra[time_delivery] = message
        self.__extra_transactions.clear()
        self.__extra_transactions.update(message)

    async def bid(
        self,
        time_delivery: TimeInterval | None = None,
        **kwargs: Any,
    ) -> None:
        """Submit a bid

        Args:
            time_delivery ([type], optional): [description]. Defaults to None.
        """

        # quantity is energy in Wh
        # price is $/kWh
        if time_delivery is None:
            time_delivery = self.__timing["next_settle"]

        entry_id = Cuid().generate(6)
        bid_entry = [
            entry_id,
            self.participant_id,
            kwargs["quantity"],  # Wh
            kwargs["price"],  # $/kWh
            time_delivery,
        ]

        self.__ledger.bids_hold[entry_id] = {
            "price": kwargs["price"],
            "quantity": kwargs["quantity"],
            "time_delivery": time_delivery,
        }
        self.__client.publish(
            f"{self.market_id}/bid",
            bid_entry,
            user_property=[("to", self.market_sid)],
            qos=1,
        )

    async def ask(
        self,
        time_delivery: TimeInterval | None = None,
        **kwargs: Any,
    ) -> None:
        """Submit an ask

        Args:
            time_delivery ([type], optional): [description]. Defaults to None.
        """
        # quantity is energy in Wh
        # price is $/kWh
        if time_delivery is None:
            time_delivery = self.__timing["next_settle"]

        entry_id = Cuid().generate(6)
        ask_entry = [
            entry_id,
            self.participant_id,
            kwargs["quantity"],  # Wh
            kwargs["price"],  # $/kWh
            time_delivery,
            kwargs["source"],
        ]

        self.__ledger.asks_hold[entry_id] = {
            "source": kwargs["source"],
            "price": kwargs["price"],
            "quantity": kwargs["quantity"],
            "time_delivery": time_delivery,
        }
        self.__client.publish(
            f"{self.market_id}/ask",
            ask_entry,
            user_property=[("to", self.market_sid)],
            qos=1,
        )

    async def ask_success(self, message: str) -> None:
        await self.__ledger.ask_success(message)

    async def bid_success(self, message: str) -> None:
        await self.__ledger.bid_success(message)

    async def settle_success(self, message: list[Any]) -> None:
        commit_id = await self.__ledger.settle_success(message)
        if commit_id:
            self.__client.publish(
                f"{self.market_id}/settlement_delivered",
                {self.participant_id: commit_id},
                user_property=[("to", self.market_sid)],
                qos=1,
            )

    async def __update_time(self, message: list[Any]) -> None:
        # synchronizes time with market
        start_time = message[0]
        duration = message[1]
        close_steps = message[2]
        end_time = start_time + duration

        self.__timing.update(
            {
                "timezone": self.timezone,
                "duration": duration,
                "last_round": (start_time - duration, start_time),
                "current_round": (start_time, end_time),
                "last_settle": (
                    start_time + duration * (close_steps - 1),
                    start_time + duration * close_steps,
                ),
                "next_settle": (
                    start_time + duration * close_steps,
                    start_time + duration * (close_steps + 1),
                ),
                "stale_round": (
                    start_time - duration * 10,
                    start_time - duration * 9,
                ),
            }
        )

    async def __update_market_info(self, message: dict[str, Any]) -> None:
        current_round_info = message.pop("current_round")
        next_settle_info = message.pop("next_settle")

        market_info = {
            self.__timing["current_round"]: {
                "grid": {
                    "buy_price": current_round_info[0],
                    "sell_price": current_round_info[1],
                }
            },
            self.__timing["next_settle"]: {
                "grid": {
                    "buy_price": next_settle_info[0],
                    "sell_price": next_settle_info[1],
                }
            },
        }

        market_info.update(message)
        self.__market_info.update(market_info)

    async def start_round(self, message: list[Any]) -> None:
        """Sequence of actions during each round
        Currently only for simulation mode.
        Real time mode needs slight modifications.

        Args:
            message ([type]): [description]
        """
        # start of current time step
        market_info = message[3]
        await self.__update_time(message)
        await self.__update_market_info(market_info)
        await self.__ledger.clear_history(self.__timing["stale_round"])
        self.__market_info.pop(self.__timing["stale_round"], None)
        # agent_act tells what actions controller should perform
        # controller should perform those actions accordingly, but it can opt out
        self.__next_actions = await self.trader.step()

        if self.action_replay is not None:
            self.__next_actions = await self.__read_actions(
                self.__timing["current_round"]
            )

        await self.__take_actions(self.__next_actions)
        # await self.trader.learn()
        if self.storage is not None:
            await self.storage.step()

        # Metering should happen at the end of the round for maximum accuracy.
        # in real-time mode there would have to be a timeout function
        # this is currently OK for simulation mode
        await self.__meter_energy(self.__timing["current_round"])
        # await self.__client.emit('end_turn', namespace='/market')
        # await self.__client.emit('end_turn')
        if self.records is not None:
            await self.records.track()
            await self.records.save(1000)
        self.__client.publish(
            f"{self.market_id}/simulation/end_turn",
            self.participant_id,
            user_property=[("to", self.market_sid)],
            qos=1,
        )

    async def make_observations_for_records(
        self, time_interval: TimeInterval
    ) -> dict[str, Any]:
        generation, consumption = await self.__read_profile(time_interval)
        net_load = consumption - generation
        obs_dict = {
            "time": str(time_interval),
            "generation": generation,
            "consumption": consumption,
            "net_load": net_load,
        }
        if self.storage is not None:
            storage_schedule = await self.storage.check_schedule(time_interval)
            obs_dict.update(storage_schedule[time_interval])

        return obs_dict

    @alru_cache
    async def __read_profile(self, time_interval: TimeInterval) -> ProfileValues:
        """Fetches energy profile for one timestamp from database

        Args:
            time_interval ([type]): [description]

        Returns:
            [type]: [description]
        """
        db = self.__profile["db"]
        table = self.__profile["db_table"]
        query = table.select().where(table.c.time == time_interval[1])
        # Direct fetch without transaction
        row = await db.fetch_one(query)
        generation, consumption = utils.process_profile(
            row=row,
            gen_scale=self.__generation_scale,
            load_scale=self.__load_scale,
        )
        return float(generation), float(consumption)

    @alru_cache
    async def __read_sensors(self, time_interval: TimeInterval) -> ProfileValues:
        """Fetches energy profile for one timestamp from database

        Args:
            time_interval ([type]): [description]

        Returns:
            [type]: [description]
        """
        db = self.__profile["db"]
        table = self.__profile["db_table"]
        query = table.select().where(table.c.time == time_interval[1])
        # Direct fetch without transaction
        row = await db.fetch_one(query)
        generation, consumption = utils.process_profile(
            row=row,
            gen_scale=self.__generation_scale,
            load_scale=self.__load_scale,
        )
        return float(generation), float(consumption)

    async def __read_actions(self, time_interval: TimeInterval) -> ActionMap:
        """Fetches energy profile for one timestamp from database

        Args:
            time_interval ([type]): [description]

        Returns:
            [type]: [description]
        """
        if self.action_replay is None:
            raise RuntimeError("Action replay is not configured.")

        db = self.action_replay["db"]
        table = self.action_replay["db_table"]
        query = table.select().where(
            (table.c.time == time_interval[1])
            & (table.c.participant_id == self.participant_id)
        )
        # Direct fetch without transaction
        row = await db.fetch_one(query)
        return cast(ActionMap, row["next_actions"])

    async def __meter_energy(self, time_interval: TimeInterval) -> bool:
        """Sends submetering data to the Market

        In simulation mode, the data is sent at the end of the current step
        In real-time mode, the data is sent at the beginning of the next step

        Args:
            time_interval ([type]): [description]

        Returns:
            [type]: [description]
        """
        if not self.server_online:
            return False

        if not self.market_connected:
            return False

        self.__meter = await self.__allocate_energy(time_interval)
        message = [self.participant_id, time_interval, self.__meter]
        self.__client.publish(
            f"{self.market_id}/meter",
            message,
            user_property=[("to", self.market_sid)],
            qos=1,
        )
        return True

    @staticmethod
    def __consume_energy(
        available: float, required: float
    ) -> tuple[float, float, float]:
        used = min(available, required)
        return available - used, required - used, used

    async def __get_settled_generation(
        self, time_interval: TimeInterval
    ) -> tuple[float, float]:
        settled_solar = 0.0
        settled_bess = 0.0
        settlements = self.__ledger.settled.get(time_interval)
        if settlements is None:
            return settled_solar, settled_bess

        for ask in settlements["asks"].values():
            if ask["source"] == "solar":
                settled_solar += ask["quantity"]
            elif ask["source"] == "bess":
                settled_bess += ask["quantity"]
            await asyncio.sleep(0)
        return settled_solar, settled_bess

    async def __get_bess_activity(
        self, time_interval: TimeInterval
    ) -> tuple[float, float]:
        if self.storage is None:
            return 0.0, 0.0

        bess_activity = await self.storage.check_schedule(time_interval)
        scheduled_energy = float(bess_activity[time_interval]["energy_scheduled"])
        bess_charge = scheduled_energy if scheduled_energy > 0 else 0.0
        bess_discharge = -scheduled_energy if scheduled_energy < 0 else 0.0
        return bess_charge, bess_discharge

    def __apply_solar_to_bess(
        self, meter: MeterData, residual_solar: float, bess_charge: float
    ) -> tuple[float, float]:
        residual_solar, bess_charge, charged_from_solar = self.__consume_energy(
            residual_solar, bess_charge
        )
        meter["load"]["bess"]["solar"] += charged_from_solar
        return residual_solar, bess_charge

    def __apply_solar_to_other_load(
        self,
        meter: MeterData,
        residual_solar: float,
        residual_consumption: float,
    ) -> tuple[float, float]:
        residual_solar, residual_consumption, solar_to_load = self.__consume_energy(
            residual_solar, residual_consumption
        )
        meter["load"]["other"]["solar"] += solar_to_load
        return residual_solar, residual_consumption

    def __apply_bess_to_other_load(
        self,
        meter: MeterData,
        bess_discharge: float,
        residual_consumption: float,
    ) -> tuple[float, float]:
        (
            bess_discharge,
            residual_consumption,
            bess_to_load,
        ) = self.__consume_energy(bess_discharge, residual_consumption)
        meter["load"]["other"]["bess"] += bess_to_load
        return bess_discharge, residual_consumption

    async def __allocate_energy(self, time_interval: TimeInterval) -> MeterData:
        """
        This function performs virtual sub metering.
        energy generated is allocated to sources by priority:
        1. settlements
        2. self consumption
        battery, then
        other loads
        3. grid
        """

        self.__timing["last_deliver"] = time_interval
        meter = self.__new_meter()
        settled_solar, _settled_bess = await self.__get_settled_generation(
            time_interval
        )
        bess_charge, bess_discharge = await self.__get_bess_activity(time_interval)
        solar_generation, residual_consumption = await self.__read_profile(
            time_interval
        )
        residual_solar = max(0.0, solar_generation - settled_solar)
        meter["generation"]["solar"] += solar_generation - residual_solar

        residual_solar, bess_charge = self.__apply_solar_to_bess(
            meter, residual_solar, bess_charge
        )
        residual_solar, residual_consumption = self.__apply_solar_to_other_load(
            meter, residual_solar, residual_consumption
        )
        bess_discharge, residual_consumption = self.__apply_bess_to_other_load(
            meter, bess_discharge, residual_consumption
        )

        meter["generation"]["solar"] += residual_solar
        meter["generation"]["bess"] += bess_discharge
        meter["load"]["other"]["ext"] += residual_consumption + bess_charge

        return meter

    async def __take_actions(self, actions: ActionMap) -> None:
        """Processes the actions given by the agent

        Args:
            actions ([type]): [description]
        """
        # actions must come in the following format:
        # actions = {
        #     'bess': {
        #         time_interval: scheduled_qty
        #     },
        #     'bids': {
        #         time_interval: {
        #             'quantity': qty,
        #             'price': dollar_per_kWh
        #         }
        #     },
        #     'asks' {
        #         source: {
        #             time_interval: {
        #                 'quantity': qty,
        #                 'price': dollar_per_kWh
        #             }
        #         }
        #     }
        # }

        # Battery charging or discharging action
        if "bess" in actions and self.storage is not None:
            for time_interval in actions["bess"]:
                await self.storage.schedule_energy(
                    actions["bess"][time_interval], ast.literal_eval(time_interval)
                )
        # Bid for energy
        if "bids" in actions:
            for time_interval in actions["bids"]:
                quantity = actions["bids"][time_interval]["quantity"]
                price = round(actions["bids"][time_interval]["price"], 4)
                await self.bid(
                    quantity=quantity,
                    price=price,
                    time_delivery=ast.literal_eval(time_interval),
                )
        # Ask to sell energy
        if "asks" in actions:
            for source in actions["asks"]:
                for time_interval in actions["asks"][source]:
                    quantity = actions["asks"][source][time_interval]["quantity"]
                    price = round(actions["asks"][source][time_interval]["price"], 4)
                    await self.ask(
                        quantity=quantity,
                        price=price,
                        source=source,
                        time_delivery=ast.literal_eval(time_interval),
                    )

    def reset(self) -> None:
        self.__ledger.reset()
        self.__extra_transactions.clear()
        self.__market_info.clear()
        self.__meter = self.__new_meter()
        self.__next_actions.clear()
        self.__timing.clear()

    async def kill(self) -> None:
        # TODO: add final actions to do for trader before killing if exists
        if hasattr(self.trader, "kill"):
            await self.trader.kill()
        await asyncio.sleep(5)
        # Close the database connection if records exist
        if self.records is not None:
            await self.records.close_connection()
        # Close the profile database connection
        if self.__profile.get("db"):
            await self.__profile["db"].disconnect()

        await self.__client.disconnect()
        os.kill(os.getpid(), signal.SIGINT)
        raise SystemExit

    async def is_participant_joined(self) -> None:
        if self.market_connected:
            self.__client.publish(
                f"{self.market_id}/simulation/participant_joined",
                self.participant_id,
                user_property=[("to", self.market_sid)],
                qos=1,
            )

    @property
    def client(self) -> MQTTClient:
        return self.__client
