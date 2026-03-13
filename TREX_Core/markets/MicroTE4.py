# ruff: noqa: N999

"""MicroTE4 market with deterministic pro-rata handling for price ties.

The market keeps standard double-auction ordering, then applies
deterministic pro-rata allocation when multiple orders share a price
level. This keeps matching fair and repeatable for identical inputs,
which is useful for reinforcement-learning workloads.
"""

from typing import Any, TypedDict, cast, override

from cuid2 import Cuid

from TREX_Core.markets.base.DoubleAuction import (
    Market as BaseMarket,
)
from TREX_Core.markets.base.DoubleAuction import (
    TimeInterval,
    TransactionRecord,
)

type PriceLevel = float | int
type OrderBook = dict[str, TransactionRecord]
type PriceLevelGroups = dict[PriceLevel, list[TransactionRecord]]
type SettlementResult = tuple[Any, Any, Any] | None


class AllocationEntry(TypedDict):
    order: TransactionRecord
    original: TransactionRecord
    quantity: float
    raw_allocation: float
    allocation: float
    frac_part: float


type AllocationMap = dict[str, AllocationEntry]


class Market(BaseMarket):
    """MicroTE4 futures market with deterministic pro-rata matching.

    Orders are grouped by delivery interval and price. Standard
    double-auction behavior is preserved, but tied price levels are
    resolved with proportional allocations so identical inputs always
    produce the same settlements.
    """

    def __init__(self, market_id, **kwargs):
        super().__init__(market_id, **kwargs)

    @override
    async def __match(self, time_delivery: TimeInterval) -> None:
        """Match open orders for a delivery interval."""
        order_books = self.__open_books_for_delivery(time_delivery)
        if order_books is None:
            return

        bids, asks = order_books
        bid_price_groups = self.__group_orders_by_price(bids)
        ask_price_groups = self.__group_orders_by_price(asks)

        for bid_price in sorted(bid_price_groups, reverse=True):
            bids_at_price = bid_price_groups[bid_price]

            for ask_price in sorted(ask_price_groups):
                if ask_price > bid_price:
                    break

                bids_at_price = await self.__process_price_pair(
                    bid_price,
                    ask_price,
                    bid_price_groups,
                    ask_price_groups,
                    bids,
                    asks,
                    time_delivery,
                )
                if not bids_at_price:
                    break

    def __open_books_for_delivery(
        self, time_delivery: TimeInterval
    ) -> tuple[OrderBook, OrderBook] | None:
        open_orders = self.__open.get(time_delivery)
        if open_orders is None or {"ask", "bid"} > open_orders.keys():
            return None

        return (
            cast(OrderBook, open_orders["bid"]),
            cast(OrderBook, open_orders["ask"]),
        )

    def __group_orders_by_price(self, orders: OrderBook) -> PriceLevelGroups:
        price_groups: PriceLevelGroups = {}
        for order in orders.values():
            price = cast(PriceLevel, order["price"])
            price_groups.setdefault(price, []).append(order)
        return price_groups

    async def __process_price_pair(
        self,
        bid_price: PriceLevel,
        ask_price: PriceLevel,
        bid_price_groups: PriceLevelGroups,
        ask_price_groups: PriceLevelGroups,
        bids: OrderBook,
        asks: OrderBook,
        time_delivery: TimeInterval,
    ) -> list[TransactionRecord]:
        bids_at_price = bid_price_groups[bid_price]
        asks_at_price = ask_price_groups[ask_price]

        if len(bids_at_price) == 1 and len(asks_at_price) == 1:
            await self.__match_single_pair(
                bids_at_price[0],
                asks_at_price[0],
                bids,
                asks,
                time_delivery,
            )
        else:
            await self.__match_pro_rata(
                bids_at_price,
                asks_at_price,
                bids,
                asks,
                time_delivery,
            )
            self.__sync_group_quantities(bids_at_price, bids)
            self.__sync_group_quantities(asks_at_price, asks)

        updated_bids = self.__active_orders(bids_at_price)
        updated_asks = self.__active_orders(asks_at_price)
        bid_price_groups[bid_price] = updated_bids
        ask_price_groups[ask_price] = updated_asks
        return updated_bids

    async def __match_single_pair(
        self,
        bid: TransactionRecord,
        ask: TransactionRecord,
        bids: OrderBook,
        asks: OrderBook,
        time_delivery: TimeInterval,
    ) -> None:
        if (
            bid["participant_id"] == ask["participant_id"]
            or bid["quantity"] <= 0
            or ask["quantity"] <= 0
        ):
            return

        bid_id = cast(str, bid["id"])
        ask_id = cast(str, ask["id"])
        original_bid = bids[bid_id]
        original_ask = asks[ask_id]
        quantity = min(
            cast(float, original_bid["quantity"]),
            cast(float, original_ask["quantity"]),
        )
        if quantity <= 0:
            return

        await self.settle(original_bid, original_ask, time_delivery)
        bid["quantity"] = original_bid["quantity"]
        ask["quantity"] = original_ask["quantity"]

    def __sync_group_quantities(
        self,
        orders_at_price: list[TransactionRecord],
        order_map: OrderBook,
    ) -> None:
        for order in orders_at_price:
            order_id = cast(str, order["id"])
            order["quantity"] = order_map[order_id]["quantity"]

    def __active_orders(
        self, orders_at_price: list[TransactionRecord]
    ) -> list[TransactionRecord]:
        return [order for order in orders_at_price if order["quantity"] > 0]

    async def __match_pro_rata(
        self,
        bids_at_price: list[TransactionRecord],
        asks_at_price: list[TransactionRecord],
        bid_map: OrderBook,
        ask_map: OrderBook,
        time_delivery: TimeInterval,
    ) -> None:
        """Match tied price levels with deterministic pro-rata allocation."""
        valid_bids = self.__valid_orders(bids_at_price, bid_map)
        valid_asks = self.__valid_orders(asks_at_price, ask_map)
        if not valid_bids or not valid_asks:
            return

        total_bid_qty = self.__total_order_quantity(valid_bids, bid_map)
        total_ask_qty = self.__total_order_quantity(valid_asks, ask_map)
        match_qty = min(total_bid_qty, total_ask_qty)
        if match_qty <= 0:
            return

        bid_allocations = self.__build_allocations(
            valid_bids,
            bid_map,
            match_qty,
            total_bid_qty,
        )
        ask_allocations = self.__build_allocations(
            valid_asks,
            ask_map,
            match_qty,
            total_ask_qty,
        )

        self.__reconcile_allocations(bid_allocations, match_qty)
        self.__reconcile_allocations(ask_allocations, match_qty)
        await self.__settle_allocations(
            bid_allocations,
            ask_allocations,
            time_delivery,
        )

    def __valid_orders(
        self,
        orders_at_price: list[TransactionRecord],
        order_map: OrderBook,
    ) -> list[TransactionRecord]:
        valid_orders: list[TransactionRecord] = []
        for order in orders_at_price:
            if order["quantity"] <= 0:
                continue

            order_id = cast(str, order["id"])
            original_order = order_map[order_id]
            if original_order["quantity"] <= 0:
                continue

            valid_orders.append(order)

        return valid_orders

    def __total_order_quantity(
        self,
        orders: list[TransactionRecord],
        order_map: OrderBook,
    ) -> float:
        return sum(
            cast(float, order_map[cast(str, order["id"])]["quantity"])
            for order in orders
        )

    def __build_allocations(
        self,
        orders: list[TransactionRecord],
        order_map: OrderBook,
        match_qty: float,
        total_qty: float,
    ) -> AllocationMap:
        allocations: AllocationMap = {}
        for order in orders:
            order_id = cast(str, order["id"])
            original_order = order_map[order_id]
            order_qty = cast(float, original_order["quantity"])
            raw_allocation = match_qty * (order_qty / total_qty)
            allocation = float(round(raw_allocation))

            allocations[order_id] = {
                "order": order,
                "original": original_order,
                "quantity": order_qty,
                "raw_allocation": raw_allocation,
                "allocation": allocation,
                "frac_part": abs(raw_allocation - allocation),
            }

        return allocations

    def __reconcile_allocations(
        self,
        allocations: AllocationMap,
        match_qty: float,
    ) -> None:
        self.__fix_quantity_conservation(allocations, match_qty)
        self.__apply_minimum_allocations(allocations, match_qty)
        self.__trim_excess_allocations(allocations, match_qty)

    def __fix_quantity_conservation(
        self,
        allocations: AllocationMap,
        match_qty: float,
    ) -> None:
        total = self.__allocation_total(allocations)
        if total == match_qty or not allocations:
            return

        sorted_allocations = sorted(
            allocations.values(),
            key=self.__conservation_sort_key,
        )
        adjustment = -1.0 if total > match_qty else 1.0
        sorted_allocations[0]["allocation"] += adjustment

    def __apply_minimum_allocations(
        self,
        allocations: AllocationMap,
        match_qty: float,
    ) -> None:
        remaining_qty = match_qty - self.__allocation_total(allocations)
        if remaining_qty <= 0:
            return

        zero_allocated = [
            allocation
            for allocation in allocations.values()
            if allocation["raw_allocation"] > 0 and allocation["allocation"] == 0
        ]
        zero_allocated.sort(
            key=lambda allocation: allocation["raw_allocation"],
            reverse=True,
        )

        for allocation in zero_allocated:
            if remaining_qty <= 0:
                break

            allocation["allocation"] = 1.0
            remaining_qty -= 1.0

    def __trim_excess_allocations(
        self,
        allocations: AllocationMap,
        match_qty: float,
    ) -> None:
        total = self.__allocation_total(allocations)
        if total <= match_qty:
            return

        excess = total - match_qty
        for allocation in sorted(allocations.values(), key=self.__excess_sort_key):
            if excess <= 0:
                break

            reduction = 0.0
            if allocation["allocation"] > 1:
                reduction = min(excess, allocation["allocation"] - 1)

            if reduction <= 0:
                continue

            allocation["allocation"] -= reduction
            excess -= reduction

    async def __settle_allocations(
        self,
        bid_allocations: AllocationMap,
        ask_allocations: AllocationMap,
        time_delivery: TimeInterval,
    ) -> None:
        for bid_data in bid_allocations.values():
            bid_allocation = bid_data["allocation"]
            if bid_allocation <= 0:
                continue

            for ask_data in ask_allocations.values():
                if bid_allocation <= 0:
                    break

                ask_allocation = ask_data["allocation"]
                if ask_allocation <= 0:
                    continue

                if (
                    bid_data["order"]["participant_id"]
                    == ask_data["order"]["participant_id"]
                ):
                    continue

                pair_qty = min(bid_allocation, ask_allocation)
                if pair_qty <= 0:
                    continue

                await self.settle(
                    bid_data["original"],
                    ask_data["original"],
                    time_delivery,
                    pair_qty,
                )
                bid_allocation -= pair_qty
                ask_data["allocation"] -= pair_qty

            bid_data["allocation"] = bid_allocation

    def __allocation_total(self, allocations: AllocationMap) -> float:
        return sum(allocation["allocation"] for allocation in allocations.values())

    @staticmethod
    def __conservation_sort_key(
        allocation: AllocationEntry,
    ) -> tuple[float, float, float]:
        return (
            -allocation["frac_part"],
            allocation["raw_allocation"],
            allocation["quantity"],
        )

    @staticmethod
    def __excess_sort_key(
        allocation: AllocationEntry,
    ) -> tuple[float, float]:
        return (-allocation["allocation"], -allocation["raw_allocation"])

    @override
    async def settle(
        self,
        bid: TransactionRecord,
        ask: TransactionRecord,
        time_delivery: TimeInterval,
        quantity_to_settle: float | None = None,
    ) -> SettlementResult:
        """Perform settlement for a matched bid and ask.

        When settlement succeeds, order quantities are updated, a
        commitment record is stored, and both participants receive a
        settlement notification.

        Parameters
        ----------
        bid : TransactionRecord
            Open bid to be settled.
        ask : TransactionRecord
            Open ask to be settled.
        time_delivery : TimeInterval
            Delivery interval in UNIX timestamp format.
        quantity_to_settle : float, optional
            Explicit quantity to settle. When omitted, the method uses
            `min(bid["quantity"], ask["quantity"])`.
        """
        if ask["source"] == "grid":
            return None

        quantity = quantity_to_settle
        if quantity is None:
            quantity = min(
                cast(float, bid["quantity"]),
                cast(float, ask["quantity"]),
            )

        if quantity <= 0:
            return None

        commit_id = Cuid().generate(6)
        settlement_time = self.__timing["current_round"][1]
        settlement_price_sell = ask["price"]
        settlement_price_buy = bid["price"]
        record = {
            "quantity": quantity,
            "seller_id": ask["participant_id"],
            "buyer_id": bid["participant_id"],
            "energy_source": ask["source"],
            "settlement_price_sell": settlement_price_sell,
            "settlement_price_buy": settlement_price_buy,
            "time_purchase": settlement_time,
        }

        if time_delivery not in self.__settled:
            self.__settled[time_delivery] = {}

        self.__settled[time_delivery][commit_id] = {
            "time_settlement": settlement_time,
            "source": ask["source"],
            "record": record,
            "ask": ask,
            "seller_id": ask["participant_id"],
            "bid": bid,
            "buyer_id": bid["participant_id"],
        }

        settled_record = self.__settled[time_delivery][commit_id]["record"]
        buyer_message = [commit_id, bid["id"], ask["source"], quantity, time_delivery]
        seller_message = [commit_id, ask["id"], ask["source"], quantity, time_delivery]

        self.__client.publish(
            f"{self.market_id}/{bid['participant_id']}/settled",
            buyer_message,
            user_property=[("to", self.__participants[bid["participant_id"]]["sid"])],
            qos=2,
        )
        self.__client.publish(
            f"{self.market_id}/{ask['participant_id']}/settled",
            seller_message,
            user_property=[("to", self.__participants[ask["participant_id"]]["sid"])],
            qos=2,
        )

        bid["quantity"] = max(0, bid["quantity"] - settled_record["quantity"])
        ask["quantity"] = max(0, ask["quantity"] - settled_record["quantity"])
        self.__status["round_settled"].append(commit_id)
        return quantity, settlement_price_buy, settlement_price_sell
