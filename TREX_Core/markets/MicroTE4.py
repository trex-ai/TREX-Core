import itertools
from cuid2 import Cuid
from operator import itemgetter
from typing import override
from TREX_Core.markets.base.DoubleAuction import Market as BaseMarket

"""
MicroTE4 Market - Deterministic Pro-Rata Double Auction

This implementation extends the standard double auction with deterministic pro-rata matching 
for price-level ties. The market mechanism is specifically designed to ensure consistent,
fair matching results while maintaining the core properties of the double auction.

Key Features:
------------
1. Standard Double Auction Behavior:
   - Bids sorted by price (highest first)
   - Asks sorted by price (lowest first)
   - Highest bids matched with lowest asks
   - Early termination when ask price > bid price

2. Pro-Rata Matching for Ties:
   - When multiple bids/asks exist at the same price level, quantities are allocated proportionally
   - Each order receives a share based on its size relative to the total
   - Example: If 100 units are to be matched among bids of [60, 40] units, 
     they receive [60, 40] units respectively

3. Deterministic Allocation:
   - Proportional allocation with rounding to nearest integer
   - Uses multiple criteria for deterministic adjustment:
     a. Fractional part of allocation (higher first)
     b. Raw allocation amount
     c. Original quantity
   - Ensures identical inputs always produce identical outputs

4. Quantity Conservation:
   - Total matched quantity is always conserved
   - Adjustments to rounded allocations maintain conservation
   - Minimum allocations only applied if conservation permits

5. Minimum Allocation Policy:
   - Orders with positive raw allocation attempt to receive at least 1 unit
   - Applied only when sufficient quantity is available
   - Quantity conservation takes precedence

Algorithm Flow:
--------------
1. Group bids and asks by price level
2. Process price levels in order (highest bids, lowest asks)
3. For non-tie cases (single bid, single ask):
   - Use standard matching with min(bid_qty, ask_qty)
4. For tie cases (multiple bids or asks at same price):
   - Calculate total quantities and match amount
   - Calculate proportional allocations
   - Apply deterministic rounding
   - Ensure quantity conservation
   - Apply minimum allocations where possible
   - Create settlements with specific quantities
5. Continue with remaining quantities in standard matching loop

This design ensures fair treatment of all orders while providing deterministic behavior
that is crucial for reinforcement learning applications. The algorithm carefully balances
market efficiency, fairness, and computational complexity.
"""


class Market(BaseMarket):
    """MicroTE4 is a futures trading based market design for transactive energy as part of TREX

    This implementation extends the standard double auction with deterministic pro-rata matching
    for price-level ties, which is critical for reinforcement learning applications.

    The market mechanism works like standard futures contracts, where delivery time interval
    is submitted along with the bid or ask. Bids and asks are organized by source type and
    time slot.

    When multiple bids or asks exist at the same price level, quantities are allocated
    proportionally based on order size. This ensures fair treatment of all orders
    and deterministic behavior for identical inputs.

    Bids/asks can be accepted for any time slot starting from one step into the future to infinity.
    The minimum close slot is determined by 'close_steps', where a close_steps of 2 is 1 step into the future.
    The minimum close time slot is the last delivery slot that will accept bids/asks.
    """

    def __init__(self, market_id, **kwargs):
        super().__init__(market_id, **kwargs)

    @override
    async def __match(self, time_delivery):
        """Pro-rata matching algorithm with deterministic allocation for ties.

        This algorithm maintains the core behavior of matching highest bids with lowest asks,
        but adds special handling for ties at the same price level. When multiple bids or asks
        exist at the same price, quantities are allocated proportionally based on order size.

        Parameters
        ----------
        time_delivery : tuple
            Tuple containing the start and end timestamps in UNIX timestamp format
            indicating the interval for energy to be delivered.
        """
        if time_delivery not in self._Market__open:
            return

        if {'ask', 'bid'} > self._Market__open[time_delivery].keys():
            return

        # Group bids and asks by price
        bids = sorted([bid.copy() for bid in self._Market__open[time_delivery]['bid'] if bid['quantity'] > 0],
                      key=itemgetter('price'), reverse=True)
        asks = sorted([ask.copy() for ask in self._Market__open[time_delivery]['ask'] if ask['quantity'] > 0],
                      key=itemgetter('price'), reverse=False)

        if not bids or not asks:
            return

        # Group bids and asks by price level to detect ties
        bid_price_groups = {}
        for bid in bids:
            if bid['price'] not in bid_price_groups:
                bid_price_groups[bid['price']] = []
            bid_price_groups[bid['price']].append(bid)

        ask_price_groups = {}
        for ask in asks:
            if ask['price'] not in ask_price_groups:
                ask_price_groups[ask['price']] = []
            ask_price_groups[ask['price']].append(ask)

        # Map from working copies back to original objects for settlement
        bid_map = {b['id']: next(original for original in self._Market__open[time_delivery]['bid']
                                 if original['id'] == b['id'])
                   for b in bids}
        ask_map = {a['id']: next(original for original in self._Market__open[time_delivery]['ask']
                                 if original['id'] == a['id'])
                   for a in asks}

        # Process each bid price group in descending order
        for bid_price in sorted(bid_price_groups.keys(), reverse=True):
            bids_at_price = bid_price_groups[bid_price]

            # Process each ask price group in ascending order
            for ask_price in sorted(ask_price_groups.keys()):
                # Early termination - if ask price exceeds bid price, no more matches possible
                if ask_price > bid_price:
                    break

                asks_at_price = ask_price_groups[ask_price]

                # Check for ties - if no ties, use standard matching
                if len(bids_at_price) == 1 and len(asks_at_price) == 1:
                    bid = bids_at_price[0]
                    ask = asks_at_price[0]

                    # Skip self-trades
                    if bid['participant_id'] == ask['participant_id'] or bid['quantity'] <= 0 or ask['quantity'] <= 0:
                        continue

                    original_bid = bid_map[bid['id']]
                    original_ask = ask_map[ask['id']]

                    # Calculate quantity from original objects
                    quantity = min(original_bid['quantity'], original_ask['quantity'])

                    if quantity > 0:
                        # Use the standard settle method
                        await self.settle(original_bid, original_ask, time_delivery)

                        # Update working copies
                        bid['quantity'] = original_bid['quantity']
                        ask['quantity'] = original_ask['quantity']
                else:
                    # We have ties, use pro-rata matching
                    await self.__match_pro_rata(bids_at_price, asks_at_price, bid_map, ask_map, time_delivery)

                    # Update working copies after pro-rata matching
                    for bid in bids_at_price:
                        original_bid = bid_map[bid['id']]
                        bid['quantity'] = original_bid['quantity']

                    for ask in asks_at_price:
                        original_ask = ask_map[ask['id']]
                        ask['quantity'] = original_ask['quantity']

                # Remove orders with zero quantity
                bid_price_groups[bid_price] = [b for b in bids_at_price if b['quantity'] > 0]
                bids_at_price = bid_price_groups[bid_price]

                ask_price_groups[ask_price] = [a for a in asks_at_price if a['quantity'] > 0]
                asks_at_price = ask_price_groups[ask_price]

                # If all bids at this price are fulfilled, move to next price
                if not bids_at_price:
                    break

    async def __match_pro_rata(self, bids_at_price, asks_at_price, bid_map, ask_map, time_delivery):
        """Match orders with pro-rata allocation for ties.

        This method handles the case where multiple bids and/or asks exist at the same price level.
        Quantities are allocated proportionally based on order size, ensuring deterministic behavior.

        Parameters
        ----------
        bids_at_price : list
            List of bids at the same price level
        asks_at_price : list
            List of asks at the same price level
        bid_map : dict
            Map from working copies to original bid objects
        ask_map : dict
            Map from working copies to original ask objects
        time_delivery : tuple
            Tuple containing delivery time interval
        """
        # Filter out orders with zero quantities and self-trades
        valid_bids = []
        for bid in bids_at_price:
            if bid['quantity'] > 0:
                original_bid = bid_map[bid['id']]
                if original_bid['quantity'] > 0:
                    valid_bids.append(bid)

        valid_asks = []
        for ask in asks_at_price:
            if ask['quantity'] > 0:
                original_ask = ask_map[ask['id']]
                if original_ask['quantity'] > 0:
                    valid_asks.append(ask)

        if not valid_bids or not valid_asks:
            return

        # Calculate total quantities
        total_bid_qty = sum(bid_map[bid['id']]['quantity'] for bid in valid_bids)
        total_ask_qty = sum(ask_map[ask['id']]['quantity'] for ask in valid_asks)
        match_qty = min(total_bid_qty, total_ask_qty)

        if match_qty <= 0:
            return

        # Calculate proportional allocations for bids
        bid_allocations = {}
        for bid in valid_bids:
            original_bid = bid_map[bid['id']]
            bid_qty = original_bid['quantity']

            # Calculate raw pro-rata allocation
            raw_allocation = match_qty * (bid_qty / total_bid_qty)

            # Initial allocation rounded to nearest integer
            allocation = round(raw_allocation)

            # Store data for allocation adjustments
            bid_allocations[bid['id']] = {
                'bid': bid,
                'original': original_bid,
                'quantity': bid_qty,
                'raw_allocation': raw_allocation,
                'allocation': allocation,
                'frac_part': abs(raw_allocation - allocation)
            }

        # Calculate proportional allocations for asks
        ask_allocations = {}
        for ask in valid_asks:
            original_ask = ask_map[ask['id']]
            ask_qty = original_ask['quantity']

            # Calculate raw pro-rata allocation
            raw_allocation = match_qty * (ask_qty / total_ask_qty)

            # Initial allocation rounded to nearest integer
            allocation = round(raw_allocation)

            # Store data for allocation adjustments
            ask_allocations[ask['id']] = {
                'ask': ask,
                'original': original_ask,
                'quantity': ask_qty,
                'raw_allocation': raw_allocation,
                'allocation': allocation,
                'frac_part': abs(raw_allocation - allocation)
            }

        # Fix quantity conservation for bids
        bid_total = sum(bid_data['allocation'] for bid_data in bid_allocations.values())
        if bid_total != match_qty:
            # Sort by multiple criteria for deterministic selection
            sorted_bids = sorted(bid_allocations.values(), key=lambda x: (
                -x['frac_part'],  # Higher fractional part first
                x['raw_allocation'],  # Then by raw allocation
                x['quantity']  # Then by original quantity
            ))

            adjustment = -1 if bid_total > match_qty else 1
            sorted_bids[0]['allocation'] += adjustment

        # Fix quantity conservation for asks
        ask_total = sum(ask_data['allocation'] for ask_data in ask_allocations.values())
        if ask_total != match_qty:
            sorted_asks = sorted(ask_allocations.values(), key=lambda x: (
                -x['frac_part'],
                x['raw_allocation'],
                x['quantity']
            ))

            adjustment = -1 if ask_total > match_qty else 1
            sorted_asks[0]['allocation'] += adjustment

        # Handle minimum allocations
        # First see how many units we could allocate after standard rounding
        bid_total = sum(bid_data['allocation'] for bid_data in bid_allocations.values())
        remaining_bid_qty = match_qty - bid_total

        # Find bids with positive raw allocation but zero rounded allocation
        zero_allocated_bids = [b for b in bid_allocations.values()
                               if b['raw_allocation'] > 0 and b['allocation'] == 0]

        # Sort by raw allocation (highest first) and allocate minimum of 1 where possible
        if remaining_bid_qty > 0 and zero_allocated_bids:
            zero_allocated_bids.sort(key=lambda x: x['raw_allocation'], reverse=True)
            for bid_data in zero_allocated_bids:
                if remaining_bid_qty > 0:
                    bid_data['allocation'] = 1
                    remaining_bid_qty -= 1
                else:
                    break

        # Do the same for asks
        ask_total = sum(ask_data['allocation'] for ask_data in ask_allocations.values())
        remaining_ask_qty = match_qty - ask_total

        zero_allocated_asks = [a for a in ask_allocations.values()
                               if a['raw_allocation'] > 0 and a['allocation'] == 0]

        if remaining_ask_qty > 0 and zero_allocated_asks:
            zero_allocated_asks.sort(key=lambda x: x['raw_allocation'], reverse=True)
            for ask_data in zero_allocated_asks:
                if remaining_ask_qty > 0:
                    ask_data['allocation'] = 1
                    remaining_ask_qty -= 1
                else:
                    break

        # Final check for quantity conservation after minimum allocations
        bid_total = sum(bid_data['allocation'] for bid_data in bid_allocations.values())
        if bid_total > match_qty:
            # If minimums caused excess, reduce from largest allocations
            sorted_bids = sorted(bid_allocations.values(), key=lambda x: (-x['allocation'], -x['raw_allocation']))
            excess = bid_total - match_qty

            for bid_data in sorted_bids:
                reduction = min(excess, bid_data['allocation'] - 1) if bid_data['allocation'] > 1 else 0
                if reduction > 0:
                    bid_data['allocation'] -= reduction
                    excess -= reduction
                    if excess == 0:
                        break

        ask_total = sum(ask_data['allocation'] for ask_data in ask_allocations.values())
        if ask_total > match_qty:
            sorted_asks = sorted(ask_allocations.values(), key=lambda x: (-x['allocation'], -x['raw_allocation']))
            excess = ask_total - match_qty

            for ask_data in sorted_asks:
                reduction = min(excess, ask_data['allocation'] - 1) if ask_data['allocation'] > 1 else 0
                if reduction > 0:
                    ask_data['allocation'] -= reduction
                    excess -= reduction
                    if excess == 0:
                        break

        # Create settlements based on allocations
        # For each bid-ask pair that doesn't involve self-trading
        for bid_id, bid_data in bid_allocations.items():
            if bid_data['allocation'] <= 0:
                continue

            original_bid = bid_data['original']
            bid_allocation = bid_data['allocation']

            for ask_id, ask_data in ask_allocations.items():
                if ask_data['allocation'] <= 0:
                    continue

                # Skip self-trading
                if bid_data['bid']['participant_id'] == ask_data['ask']['participant_id']:
                    continue

                original_ask = ask_data['original']
                ask_allocation = ask_data['allocation']

                # Calculate pair quantity
                pair_qty = min(bid_allocation, ask_allocation)

                if pair_qty > 0:
                    # Create settlement with explicit quantity
                    await self.settle(original_bid, original_ask, time_delivery, pair_qty)

                    # Update remaining allocations
                    bid_allocation -= pair_qty
                    ask_data['allocation'] -= pair_qty

                    if bid_allocation <= 0:
                        break

    @override
    async def settle(self, bid: dict, ask: dict, time_delivery: tuple, quantity_to_settle=None):
        """Performs settlement for bid/ask pairs found during the matching process.

        If bid/ask are valid, the bid/ask quantities are adjusted, a commitment record is created, and a settlement confirmation is sent to both participants.

        Parameters
        ----------
        bid: dict
            bid entry to be settled. Should be a reference to the open bid

        ask: dict
            bid entry to be settled. Should be a reference to the open ask

        time_delivery : tuple
            Tuple containing the start and end timestamps in UNIX timestamp format.

        quantity_to_settle : float, optional
            If provided, uses this exact value for settlement instead of calculating min(bid['quantity'], ask['quantity'])
            This is useful for pro-rata matching where quantities are pre-calculated.

        locking: bool
        Optinal locking mode, which locks the bid and ask until a callback is received after settlement confirmation is sent. The default value is False.

        Currently, locking should be disabled in simulation mode, as waiting for callback causes some settlements to be incomplete, likely due a flaw in the implementation or a poor understanding of how callbacks affect the sequence of events to be executed in async mode.

        Notes
        -----
        It is possible to settle directly with the grid, although this feature is currently not used by the agents and is under consideration to be deprecated.


        """

        # grid is not allowed to interact through market
        if ask['source'] == 'grid':
            return

        # Use specified quantity if provided, otherwise calculate as min
        quantity = quantity_to_settle if quantity_to_settle is not None else min(bid['quantity'], ask['quantity'])

        # only proceed to settle if settlement quantity is positive
        if quantity <= 0:
            return

        # if locking:
        #     # lock the bid and ask until confirmations are received
        #     ask['lock'] = True
        #     bid['lock'] = True

        commit_id = Cuid().generate(6)
        settlement_time = self.__timing['current_round'][1]
        settlement_price_sell = ask['price']
        settlement_price_buy = bid['price']
        record = {
            'quantity': quantity,
            'seller_id': ask['participant_id'],
            'buyer_id': bid['participant_id'],
            'energy_source': ask['source'],
            'settlement_price_sell': settlement_price_sell,
            'settlement_price_buy': settlement_price_buy,
            'time_purchase': settlement_time
        }

        # Record successful settlements
        if time_delivery not in self.__settled:
            self.__settled[time_delivery] = {}

        self.__settled[time_delivery][commit_id] = {
            'time_settlement': settlement_time,
            'source': ask['source'],
            'record': record,
            'ask': ask,
            'seller_id': ask['participant_id'],
            'bid': bid,
            'buyer_id': bid['participant_id'],
        }

        # if buyer == 'grid' or seller == 'grid':
        # if buy_price is not None and sell_price is not None:
        #     return
        buyer_message = [
            commit_id,
            bid['id'],
            ask['source'],
            quantity,
            time_delivery
        ]

        seller_message = [
            commit_id,
            ask['id'],
            ask['source'],
            quantity,
            time_delivery
        ]
        self.__client.publish(f'{self.market_id}/{bid['participant_id']}/settled', buyer_message,
                              user_property=('to', self.__participants[bid['participant_id']]['sid']), qos=0)
        self.__client.publish(f'{self.market_id}/{ask['participant_id']}/settled', seller_message,
                              user_property=('to', self.__participants[ask['participant_id']]['sid']), qos=0)
        bid['quantity'] = max(0, bid['quantity'] - self.__settled[time_delivery][commit_id]['record']['quantity'])
        ask['quantity'] = max(0, ask['quantity'] - self.__settled[time_delivery][commit_id]['record']['quantity'])
        self.__status['round_settled'].append(commit_id)
        return quantity, settlement_price_buy, settlement_price_sell

    # @override
    # async def __match(self, time_delivery):
    #     """Matches bids with asks for a single source type in a time slot
    #
    #     THe matching and settlement process closely resemble double auctions.
    #     For all bids/asks for a source in the delivery time slots, highest bids are matched with lowest asks
    #     and settled pairwise. Quantities can be partially settled. Unsettled quantities are discarded. Participants are only obligated to buy/sell quantities settled for the delivery period.
    #
    #     Parameters
    #     ----------
    #     time_delivery : tuple
    #         Tuple containing the start and end timestamps in UNIX timestamp format indicating the interval for energy to be delivered.
    #
    #     Notes
    #     -----
    #     Presently, the settlement price is hard-coded as the average price of the bid/ask pair. In the near future, dedicated, more sophisticated functions for determining settlement price will be implemented
    #
    #     """
    #
    #     if time_delivery not in self.__open:
    #         return
    #
    #     if 'ask' not in self.__open[time_delivery]:
    #         return
    #
    #     if 'bid' not in self.__open[time_delivery]:
    #         return
    #
    #     # remove zero-quantity bid and ask entries
    #     # sort bids by decreasing price and asks by increasing price
    #     # def filter_bids_asks():
    #     self.__open[time_delivery]['ask'][:] = \
    #         sorted([ask for ask in self.__open[time_delivery]['ask'] if ask['quantity'] > 0],
    #                key=itemgetter('price'), reverse=False)
    #     self.__open[time_delivery]['bid'][:] = \
    #         sorted([bid for bid in self.__open[time_delivery]['bid'] if bid['quantity'] > 0],
    #                key=itemgetter('price'), reverse=True)
    #
    #     # await asyncio.get_event_loop().run_in_executor(filter_bids_asks)
    #
    #     bids = self.__open[time_delivery]['bid']
    #     asks = self.__open[time_delivery]['ask']
    #
    #     for bid, ask, in itertools.product(bids, asks):
    #         if ask['price'] > bid['price']:
    #             continue
    #
    #         if bid['participant_id'] == ask['participant_id']:
    #             continue
    #
    #         if bid['quantity'] <= 0 or ask['quantity'] <= 0:
    #             continue
    #         await self.__settle(bid, ask, time_delivery)

    # @override
    # async def __settle(self, bid: dict, ask: dict, time_delivery: tuple):
    #     """Performs settlement for bid/ask pairs found during the matching process.
    #
    #     If bid/ask are valid, the bid/ask quantities are adjusted, a commitment record is created, and a settlement confirmation is sent to both participants.
    #
    #     Parameters
    #     ----------
    #     bid: dict
    #         bid entry to be settled. Should be a reference to the open bid
    #
    #     ask: dict
    #         bid entry to be settled. Should be a reference to the open ask
    #
    #     time_delivery : tuple
    #         Tuple containing the start and end timestamps in UNIX timestamp format.
    #
    #     locking: bool
    #     Optinal locking mode, which locks the bid and ask until a callback is received after settlement confirmation is sent. The default value is False.
    #
    #     Currently, locking should be disabled in simulation mode, as waiting for callback causes some settlements to be incomplete, likely due a flaw in the implementation or a poor understanding of how callbacks affect the sequence of events to be executed in async mode.
    #
    #     Notes
    #     -----
    #     It is possible to settle directly with the grid, although this feature is currently not used by the agents and is under consideration to be deprecated.
    #
    #
    #     """
    #
    #     # grid is not allowed to interact through market
    #     if ask['source'] == 'grid':
    #         return
    #
    #     # only proceed to settle if settlement quantity is positive
    #     quantity = min(bid['quantity'], ask['quantity'])
    #     if quantity <= 0:
    #         return
    #
    #     # if locking:
    #     #     # lock the bid and ask until confirmations are received
    #     #     ask['lock'] = True
    #     #     bid['lock'] = True
    #
    #     commit_id = Cuid().generate(6)
    #     settlement_time = self.__timing['current_round'][1]
    #     settlement_price_sell = ask['price']
    #     settlement_price_buy = bid['price']
    #     record = {
    #         'quantity': quantity,
    #         'seller_id': ask['participant_id'],
    #         'buyer_id': bid['participant_id'],
    #         'energy_source': ask['source'],
    #         'settlement_price_sell': settlement_price_sell,
    #         'settlement_price_buy': settlement_price_buy,
    #         'time_purchase': settlement_time
    #     }
    #
    #     # Record successful settlements
    #     if time_delivery not in self.__settled:
    #         self.__settled[time_delivery] = {}
    #
    #     self.__settled[time_delivery][commit_id] = {
    #         'time_settlement': settlement_time,
    #         'source': ask['source'],
    #         'record': record,
    #         'ask': ask,
    #         'seller_id': ask['participant_id'],
    #         'bid': bid,
    #         'buyer_id': bid['participant_id'],
    #         # 'lock': locking
    #     }
    #
    #     # if buyer == 'grid' or seller == 'grid':
    #     # if buy_price is not None and sell_price is not None:
    #     #     return
    #     buyer_message = [
    #         commit_id,
    #         bid['id'],
    #         ask['source'],
    #         quantity,
    #         time_delivery
    #     ]
    #
    #     seller_message = [
    #         commit_id,
    #         ask['id'],
    #         ask['source'],
    #         quantity,
    #         time_delivery
    #     ]
    #     await self.__client.publish(bid['participant_id'], self.__participants[bid['participant_id']]['sid'],
    #                                      buyer_message)
    #     await self.__client.publish(ask['participant_id'], self.__participants[ask['participant_id']]['sid'],
    #                                      seller_message)
    #
    #     bid['quantity'] = max(0, bid['quantity'] - self.__settled[time_delivery][commit_id]['record']['quantity'])
    #     ask['quantity'] = max(0, ask['quantity'] - self.__settled[time_delivery][commit_id]['record']['quantity'])
    #     self.__status['round_settled'].append(commit_id)
