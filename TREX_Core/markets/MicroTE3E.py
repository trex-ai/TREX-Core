import itertools
from cuid2 import Cuid
from operator import itemgetter
from typing import override
from TREX_Core.markets.base.DoubleAuction import Market as BaseMarket
class Market(BaseMarket):
    """MicroTE is a futures trading based market design for transactive energy as part of TREX

    The market mechanism here works more like standard futures contracts,
    where delivery time interval is submitted along with the bid or ask.

    bids and asks are organized by source type
    the addition of delivery time requires that the bids and asks to be further organized by time slot

    Bids/asks can be are accepted for any time slot starting from one step into the future to infinity
    The minimum close slot is determined by 'close_steps', where a close_steps of 2 is 1 step into the future
    The the minimum close time slot is the last delivery slot that will accept bids/asks

    """

    def __init__(self, market_id, **kwargs):
        super().__init__(market_id, **kwargs)

    @override
    async def __match(self, time_delivery):
        """Efficient matching algorithm that reduces unnecessary comparisons.

        This is an optimized version of the matching algorithm that maintains
        identical matching behavior while reducing computational complexity.

        Key optimizations:
        1. Early termination when ask price exceeds bid price
        2. Removal of fully matched asks from consideration
        3. Early termination when a bid is fully matched

        Parameters
        ----------
        time_delivery : tuple
            Tuple containing the start and end timestamps in UNIX timestamp format
            indicating the interval for energy to be delivered.
        """
        if time_delivery not in self.__open:
            return

        if {'ask', 'bid'} > self.__open[time_delivery].keys():
            return

        # Create working copies of the bids and asks to avoid modifying the originals
        # until settlements are confirmed
        bids = sorted([bid.copy() for bid in self.__open[time_delivery]['bid'] if bid['quantity'] > 0],
                      key=itemgetter('price'), reverse=True)
        asks = sorted([ask.copy() for ask in self.__open[time_delivery]['ask'] if ask['quantity'] > 0],
                      key=itemgetter('price'), reverse=False)

        if not bids or not asks:
            return

        # Map from working copies back to original objects for settlement
        bid_map = {b['id']: next(original for original in self.__open[time_delivery]['bid']
                                 if original['id'] == b['id'])
                   for b in bids}
        ask_map = {a['id']: next(original for original in self.__open[time_delivery]['ask']
                                 if original['id'] == a['id'])
                   for a in asks}

        # For each bid, find compatible asks more efficiently
        for bid in bids:
            if bid['quantity'] <= 0:
                continue

            # For each ask, but with early stopping when no more matches possible
            ask_idx = 0
            while ask_idx < len(asks):
                ask = asks[ask_idx]

                # If we've reached an ask price higher than bid price, no more matches are possible
                if ask['price'] > bid['price']:
                    break

                # Skip self-trading and empty quantities
                if bid['participant_id'] == ask['participant_id'] or ask['quantity'] <= 0:
                    ask_idx += 1
                    continue

                # Find original objects for settlement
                original_bid = bid_map[bid['id']]
                original_ask = ask_map[ask['id']]

                # Calculate quantity using the ORIGINAL objects to ensure accuracy
                quantity = min(original_bid['quantity'], original_ask['quantity'])

                if quantity <= 0:
                    ask_idx += 1
                    continue

                # Settle using original objects
                settlement_result = await self.settle(original_bid, original_ask, time_delivery)

                # Update working copies with the current quantities from original objects
                # to keep them in sync after settlement
                bid['quantity'] = original_bid['quantity']
                ask['quantity'] = original_ask['quantity']

                # If ask is completely fulfilled, remove it to avoid future checks
                if ask['quantity'] <= 0:
                    asks.pop(ask_idx)
                else:
                    ask_idx += 1

                # If bid is completely fulfilled, move to next bid
                if bid['quantity'] <= 0:
                    break

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
