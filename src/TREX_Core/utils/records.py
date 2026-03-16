import asyncio
import datetime
from collections.abc import Awaitable, Callable
from typing import Any, NotRequired, Protocol, TypedDict, cast

import databases
import orjson
import sqlalchemy
import structlog
from sqlalchemy import Column, MetaData, Table
from sqlalchemy.engine import Engine

from TREX_Core.utils import db_utils

log = structlog.get_logger()


class ColumnConfig(TypedDict):
    type: str
    primary: NotRequired[bool]


class StorageContextLike(Protocol):
    def get_info(self, *fields: str) -> dict[str, Any]: ...


class TraderContextLike(Protocol):
    metadata: dict[str, Any] | None


class ParticipantContextLike(Protocol):
    participant_id: str
    timing: dict[str, Any]
    next_actions: Callable[[], dict[str, Any]] | None
    meter: Callable[[], dict[str, Any]] | None
    storage: StorageContextLike | None
    trader: TraderContextLike | None


class DatabaseState(TypedDict):
    path: str
    sa_engine: Engine
    connection: databases.Database | None
    table: NotRequired[Table | None]


ResultsCache = dict[str, Any]
RecordRow = dict[str, Any]
Handler = Callable[[ResultsCache], Awaitable[Any]]

SQLALCHEMY_COLUMN_TYPES: dict[str, Any] = {
    "Float": sqlalchemy.Float,
    "Integer": sqlalchemy.Integer,
    "JSON": sqlalchemy.JSON,
    "String": sqlalchemy.String,
}


class Records:
    def __init__(
        self,
        db_string: str,
        columns: list[str],
        *,
        context: ParticipantContextLike | None = None,
    ) -> None:
        self.__HANDLERS: dict[str, tuple[ColumnConfig, Handler, list[str]]] = {
            # column metadata, handler function, dependencies
            "time": (
                {"type": "Integer", "primary": True},
                self.__get_current_round,
                [],
            ),
            "participant_id": (
                {"type": "String", "primary": True},
                self.__get_participant_id,
                [],
            ),
            "meter": ({"type": "JSON"}, self.__get_meter, []),
            "next_actions": ({"type": "JSON"}, self.__get_next_actions, []),
            # 'remaining_energy':{"type": "Integer"}
            "metadata": ({"type": "JSON"}, self.__get_participant_metadata, []),
            # "next_actions": {"type": "JSON"},
            "storage_info": ({"type": "JSON"}, self.__get_storage_info, []),
            "remaining_energy": (
                {"type": "Integer"},
                self.__get_remaining_energy,
                ["storage_info"],
            ),
            "state_of_charge": (
                {"type": "Float"},
                self.__get_state_of_charge,
                ["storage_info"],
            ),
        }

        self.__db: DatabaseState = {
            "path": db_string,
            "sa_engine": sqlalchemy.create_engine(db_string),
            "connection": None,
        }
        self.__columns: dict[str, ColumnConfig] = {
            "time": {"type": "Integer", "primary": True},
            "participant_id": {"type": "String", "primary": True},
        }
        for column in columns:
            column_meta = self.__HANDLERS.get(column)
            if column_meta is not None:
                self.__columns[column] = column_meta[0]

        self.__handle_order = self.order_handler(list(self.__columns.keys()))

        self.__records: list[RecordRow] = []
        self.__last_record_time = 0.0
        self.__transactions_count = 0
        self.__meta = MetaData()

        # Track pending database write tasks
        self.__pending_write_tasks: list[asyncio.Task[None]] = []
        self.participant = context

    async def create_table(self, table_name: str) -> bool | None:
        table_name += "_records"
        columns = [
            Column(
                record,
                self.__get_column_type(self.__columns[record]["type"]),
                primary_key=self.__columns[record].get("primary", False),
            )
            for record in self.__columns
        ]
        table = sqlalchemy.Table(str(table_name), self.__meta, *columns)
        return cast(bool | None, await db_utils.create_table(self.__db["path"], table))
        # return table

    async def open_db(self, table_name: str, db_string: str | None = None) -> None:
        if db_string is None:
            db_string = self.__db["path"]
        table_name += "_records"
        self.__db["table"] = cast(
            Table | None, db_utils.get_table(db_string, table_name)
        )
        # Initialize the database connection
        connection = self.__db.get("connection")
        if connection is None:
            connection = databases.Database(db_string)
            await connection.connect()
            self.__db["connection"] = connection

    async def track(self) -> None:
        results_cache: ResultsCache = {}
        for key in self.__handle_order:
            if key in results_cache:
                continue
            results_cache[key] = await self.__HANDLERS[key][1](results_cache)
        filtered_records = {
            key: results_cache[key] for key in self.__columns if key in results_cache
        }
        self.__records.append(filtered_records)

    async def save(
        self,
        buf_len: int = 0,
        final: bool = False,
        check_table_len: bool = False,
    ) -> bool:
        """Record the buffered records into the database

        Args:
            buf_len: Minimum buffer length to trigger a write (default=0)
            final: If True, wait for the write to complete before returning
            check_table_len: If True, verify record count after write (not implemented)

        Returns:
            False if no write was performed (due to buffer conditions)
            True if a write was initiated
        """
        _ = check_table_len

        records_len = len(self.__records)
        if records_len < buf_len:
            return False

        # Swap the entire buffer instead of slicing (more efficient)
        records_to_write = self.__records
        self.__records = []  # Create a fresh list for new records

        # Create and track the database write task
        db_task = asyncio.create_task(
            db_utils.dump_data(
                records_to_write,
                self.__db["path"],
                self.__db["table"],
                existing_connection=self.__db.get("connection"),
            )
        )

        # Add to our tracking list
        self.__pending_write_tasks.append(db_task)

        # Set up callback to remove from our list when done
        def task_done_callback(completed_task: asyncio.Future[None]) -> None:
            completed_db_task = cast(asyncio.Task[None], completed_task)
            if completed_db_task in self.__pending_write_tasks:
                self.__pending_write_tasks.remove(completed_db_task)

        db_task.add_done_callback(task_done_callback)

        # For critical writes (final=True), wait for completion
        if final:
            await db_task

        self.__last_record_time = datetime.datetime.now(datetime.UTC).timestamp()
        self.__transactions_count += records_len
        return True

    async def ensure_records_complete(self) -> bool:
        """Ensure all database write tasks are complete before continuing.

        This method will:
        1. Trigger a final write of any pending records
        2. Wait for all pending database write tasks to complete
        3. Verify the record count if needed

        Returns:
            True if all records completed successfully

        Raises:
            TimeoutError: If writes don't complete within timeout period
        """
        # First do one final write and wait for it to complete
        await self.save(final=True)

        # Now wait for ALL remaining in-flight tasks
        if self.__pending_write_tasks:
            # Wait for all pending tasks to complete
            await asyncio.wait(self.__pending_write_tasks)

            # Check if we timed out and still have pending tasks
            remaining = [task for task in self.__pending_write_tasks if not task.done()]
            if remaining:
                raise TimeoutError(
                    "Timed out waiting for "
                    f"{len(remaining)} database writes to complete"
                )

        return True

    # Add method to properly close the connection when done
    async def close_connection(self) -> None:
        """Close the database connection when done"""
        # First ensure all write tasks are complete
        try:
            await self.ensure_records_complete()
        except Exception as error:
            # Log the error but continue to close the connection
            log.error("Error ensuring records complete", error=str(error))

        # Now safe to close the connection
        connection = self.__db.get("connection")
        if connection is not None:
            await connection.disconnect()
            self.__db["connection"] = None

    def order_handler(self, records: list[str]) -> list[str]:
        """
        Build snapshot dict honouring handler dependencies.
        Each handler receives the accumulating snapshot so it can consume
        results of its dependencies.
        """
        # Client always sends a `meta` dict; use its keys for requested snapshots
        seen: set[str] = set()
        order: list[str] = []

        def dfs(key: str) -> None:
            nonlocal seen, order
            if key in seen:
                return
            if key not in self.__HANDLERS:  # skip unknown base names
                return
            seen.add(key)
            deps = self.__HANDLERS[key][2]
            for dep in deps:
                dfs(dep)
            order.append(key)

        for k in records:
            dfs(k)

        return order

    def __get_column_type(self, type_name: str) -> Any:
        if type_name not in SQLALCHEMY_COLUMN_TYPES:
            raise ValueError(f"Unsupported SQLAlchemy column type: {type_name}")
        return SQLALCHEMY_COLUMN_TYPES[type_name]

    def __require_participant(self) -> ParticipantContextLike:
        if self.participant is None:
            raise RuntimeError("A participant context is required to track records.")
        return self.participant

    async def __get_current_round(self, _results_cache: ResultsCache) -> Any:
        # return 'round'
        participant = self.__require_participant()
        return participant.timing["current_round"][1]

    async def __get_participant_id(self, _results_cache: ResultsCache) -> str:
        # return 'id'
        participant = self.__require_participant()
        return participant.participant_id

    async def __get_meter(self, _results_cache: ResultsCache) -> dict[str, Any] | None:
        # return 'meter'
        participant = self.__require_participant()
        if participant.meter is None:
            return None
        return participant.meter()

    async def __get_next_actions(
        self, _results_cache: ResultsCache
    ) -> dict[str, Any] | None:
        participant = self.__require_participant()
        if participant.next_actions is None:
            return None
        return participant.next_actions()

    async def __get_storage_info(
        self, _results_cache: ResultsCache
    ) -> dict[str, Any] | None:
        participant = self.__require_participant()
        if participant.storage is None:
            return None
        return participant.storage.get_info("remaining_energy", "state_of_charge")

    async def __get_remaining_energy(self, results_cache: ResultsCache) -> Any:
        storage_info = results_cache.get("storage_info")
        if not isinstance(storage_info, dict):
            return None
        return storage_info.get("remaining_energy")

    async def __get_state_of_charge(self, results_cache: ResultsCache) -> Any:
        storage_info = results_cache.get("storage_info")
        if not isinstance(storage_info, dict):
            return None
        return storage_info.get("state_of_charge")

    async def __get_participant_metadata(
        self, _results_cache: ResultsCache
    ) -> dict[str, Any] | None:
        participant = self.__require_participant()
        trader = participant.trader
        if trader is None or trader.metadata is None:
            return None
        return cast(dict[str, Any], orjson.loads(orjson.dumps(trader.metadata)))
