import asyncio
from collections.abc import Sequence
from typing import Literal, TypedDict, cast

import structlog

logger = structlog.get_logger()

TimeInterval = tuple[int, int]
Transaction = tuple[Literal["bid", "ask"], float, float, str]


class LedgerEntry(TypedDict, total=False):
    price: float
    quantity: float
    source: str
    time_delivery: TimeInterval


class SettlementEntry(TypedDict):
    source: str
    price: float
    quantity: float


class SettlementBucket(TypedDict):
    bids: dict[str, SettlementEntry]
    asks: dict[str, SettlementEntry]


class Ledger:
    """Track accepted bids, asks, and successful settlements.

    The ledger also converts raw market records into formats used
    downstream in the participant data pipeline.
    """

    def __init__(self, participant_id: str) -> None:
        self.__participant_id = participant_id
        self.bids_hold: dict[str, LedgerEntry] = {}
        self.asks_hold: dict[str, LedgerEntry] = {}
        self.bids: dict[TimeInterval, dict[str, LedgerEntry]] = {}
        self.asks: dict[TimeInterval, dict[str, LedgerEntry]] = {}
        self.settled: dict[TimeInterval, SettlementBucket] = {}
        self.extra: dict[TimeInterval, object] = {}

    async def bid_success(self, entry_id: str) -> None:
        """Track a bid that was accepted by the market.

        Args:
            entry_id (str): The accepted bid identifier.
        """
        # TODO: may need to add an async lock here
        entry = self.bids_hold.pop(entry_id, None)
        if not entry:
            return
        time_delivery = entry["time_delivery"]
        if time_delivery not in self.bids:
            self.bids[time_delivery] = {}
        self.bids[time_delivery][entry_id] = {
            "price": entry["price"],
            "quantity": entry["quantity"],
        }

    async def ask_success(self, entry_id: str) -> None:
        """Track an ask that was accepted by the market.

        Args:
            entry_id (str): The accepted ask identifier.
        """
        # TODO: may need to add an async lock here
        entry = self.asks_hold.pop(entry_id, None)
        if not entry:
            return
        time_delivery = entry["time_delivery"]
        if time_delivery not in self.asks:
            self.asks[time_delivery] = {}
        self.asks[time_delivery][entry_id] = {
            "source": entry["source"],
            "price": entry["price"],
            "quantity": entry["quantity"],
        }

    async def settle_success(self, confirmation: Sequence[object]) -> str | None:
        """Track a successful settlement.

        Args:
            confirmation (Sequence[object]): Settlement payload published by
                the market.
        """
        # TODO: add validity checks, and feedback messages for invalid settlements
        # TODO: may need to add an async lock here

        commit_id = cast(str, confirmation[0])
        entry_id = cast(str, confirmation[1])
        source = cast(str, confirmation[2])
        quantity = cast(float, confirmation[3])
        time_delivery = cast(
            TimeInterval,
            tuple(cast(Sequence[int], confirmation[4])),
        )

        if time_delivery not in self.settled:
            self.settled[time_delivery] = {"bids": {}, "asks": {}}
        # make sure settled bid exists in local record as well
        entry_group: Literal["bids", "asks"] | None = None
        entries: dict[TimeInterval, dict[str, LedgerEntry]] | None = None

        if time_delivery in self.bids and entry_id in self.bids[time_delivery]:
            entry_group = "bids"
            entries = self.bids
        elif time_delivery in self.asks and entry_id in self.asks[time_delivery]:
            entry_group = "asks"
            entries = self.asks
        else:
            logger.warning("Invalid settlement confirmation", confirmation=confirmation)
            return None

        if commit_id in self.settled[time_delivery][entry_group]:
            return None

        entry = entries[time_delivery][entry_id]
        self.settled[time_delivery][entry_group][commit_id] = {
            "source": source,
            "price": entry["price"],
            "quantity": quantity,
        }
        # update the local entry after recording the settlement
        entry["quantity"] -= quantity
        if entry["quantity"] <= 0:
            entries[time_delivery].pop(entry_id)
        return commit_id

    async def get_settled_info(
        self,
        time_interval: TimeInterval,
    ) -> dict[str, dict[str, float]]:
        """Summarize settled data for a delivery interval.

        Args:
            time_interval (tuple): Must be one of the market rounds that
                already occurred.

        Returns:
            dict[str, dict[str, float]]: Aggregated bid and ask quantities,
                totals, and average prices.
        """
        info = {
            "asks": {"quantity": 0.0, "total_profit": 0.0, "price": 0.0},
            "bids": {"quantity": 0.0, "total_cost": 0.0, "price": 0.0},
        }

        if time_interval not in self.settled:
            return info

        settlements = self.settled[time_interval]
        for settlement in settlements["asks"].values():
            info["asks"]["quantity"] += settlement["quantity"]
            info["asks"]["total_profit"] += settlement["quantity"] * settlement["price"]

        if info["asks"]["quantity"] > 0:
            info["asks"]["price"] = (
                info["asks"]["total_profit"] / info["asks"]["quantity"]
            )

        for settlement in settlements["bids"].values():
            info["bids"]["quantity"] += settlement["quantity"]
            info["bids"]["total_cost"] += settlement["quantity"] * settlement["price"]

        if info["bids"]["quantity"] > 0:
            info["bids"]["price"] = (
                info["bids"]["total_cost"] / info["bids"]["quantity"]
            )

        return info

    async def get_simplified_transactions(
        self,
        time_interval: TimeInterval,
    ) -> list[Transaction]:
        """Return simplified transactions for a time interval.

        Args:
            time_interval (tuple): Must be one of the market rounds that
                already occurred.
        """
        transactions: list[Transaction] = []
        if time_interval not in self.settled:
            return transactions

        settlements = self.settled[time_interval]
        for action in ("bids", "asks"):
            for settlement in settlements[action].values():
                transaction_type: Literal["bid", "ask"] = (
                    "bid" if action == "bids" else "ask"
                )
                transactions.append(
                    (
                        transaction_type,
                        settlement["quantity"],
                        settlement["price"],
                        settlement["source"],
                    )
                )
            await asyncio.sleep(0)
        return transactions

    async def clear_history(self, time_interval: TimeInterval) -> None:
        self.bids_hold.clear()
        self.asks_hold.clear()
        self.bids.pop(time_interval, None)
        self.asks.pop(time_interval, None)
        self.settled.pop(time_interval, None)
        self.extra.pop(time_interval, None)

    def reset(self) -> None:
        self.bids.clear()
        self.asks.clear()
        self.settled.clear()
        self.extra.clear()
