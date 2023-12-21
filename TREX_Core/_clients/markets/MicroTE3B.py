import numpy as np
import calendar
import datetime
import time
from operator import itemgetter
import itertools
from cuid2 import Cuid as cuid
import tenacity
import os
import signal

import asyncio
from TREX_Core._clients.markets.Grid import Market as Grid
from TREX_Core._utils import db_utils, source_classifier


class Market:
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
        self.server_online = False
        self.run = True
        # Initialize timing intervals and definitions
        self.__status = {
            'active_participants': 0,
            'round_metered': 0,
            'round_matched': False,
            'round_settled': [],
            # 'round_settle_delivered': []
            'round_settle_delivered': dict()
        }

        self.__time_step_s = kwargs['time_step_size'] if 'time_step_size' in kwargs else 60
        self.__timing = {
            'mode': 'sim',
            'timezone': kwargs['timezone'],
            'current_round': (0, self.__time_step_s),
            'duration': self.__time_step_s,
            'last_round': (0, 0),
            'close_steps': kwargs['close_steps'] if 'close_steps' in kwargs else 2
            # close steps = 2 for 1 step-ahead market agent debugging

            # Ideally close_steps should be 16 for a 15-step ahead market.
            # bids and asks are settled 15 steps ahead of delivery time
            # settle takes 1 step after bid/ask submision
        }

        self.__db = {}
        self.save_transactions = True
        self.market_id = market_id
        self.__client = kwargs['sio_client']
        self.__server_ts = 0

        self.__clients = {}
        self.__participants = {}

        self.__grid = Grid(**kwargs['grid_params'])

        self.__open = {}
        self.__settled = {}
        self.__transactions = []
        self.__transaction_last_record_time = 0
        self.transactions_count = 0

    def __time(self):
        """Return time based on time convention

        Market timing operates in two modes: real-time, and simulation.
        In real-time mode, the market has control of timing, and
        in simulation mode, the simulation controller has control
        Because of this, the way time propagates through the system is slightly different between modes

        In real-time mode, master time is acquired from the system clock of the market
        In simulation mode, master time is the last time tuple that was received from the simulation controller
        """
        if self.__timing['mode'] == 'rt':
            return calendar.timegm(time.gmtime())
        if self.__timing['mode'] == 'sim':
            return self.__server_ts

    def mode_switch(self, mode):
        """Switch timing modes between real-time mode and simulation mode

        """
        self.__timing['mode'] = mode

    async def open_db(self, db_string, table_name):
        if not self.save_transactions:
            return

        self.__db['path'] = db_string
        self.__db['table_name'] = table_name

        if 'table' not in self.__db or self.__db['table'] is None:
            table_name = self.__db.pop('table_name') + '_market'
            await db_utils.create_table(
                db_string=db_string,
                table_type='market2',
                table_name=table_name)
            self.__db['table'] = db_utils.get_table(db_string, table_name)

    # async def register(self):
    #     """Function that attempts to register Market client with socket.io server in the market namespace
    #
    #     """
    #
    #     async def register_cb(success):
    #         if success:
    #             self.server_online = True
    #
    #     client_data = {
    #         'type': 'MicroTE',
    #         'id': self.market_id
    #     }
    #     await self.__client.emit('register_market', client_data, callback=register_cb)

    async def participant_connected(self, client_data):
        if client_data['id'] not in self.__participants:
            self.__participants[client_data['id']] = {
                # 'sid': client_data['sid'],
                'online': True,
                'meter': {}
            }
        else:
            # if previously registered participant returned, update with new session ID and toggle online status
            self.__participants[client_data['id']].update({
                # 'sid': client_data['sid'],
                'online': True
            })
        # self.__clients[client_data['sid']] = client_data['id']
        self.__status['active_participants'] = min(self.__status['active_participants'] + 1,
                                                   len(self.__participants))
        self.__client.publish('/'.join([self.market_id, client_data['id'], 'update_market_info']), self.market_id)
        # return self.market_id, client_data['sid']

    async def participant_disconnected(self, participant_id):
        # if a registered participant disconnects for any reason, switch online status to off
        self.__participants[participant_id].update({
            'online': False
        })
        # self.__clients.pop(self.__participants[participant_id]['sid'], None)
        self.__status['active_participants'] -= 1

    async def __classify_source(self, source):
        return await source_classifier.classify(source)

    # Initialize variables for new time step
    def __reset_status(self):
        self.__status['round_metered'] = 0
        self.__status['round_matched'] = False
        self.__status['round_settled'].clear()
        self.__status['round_settle_delivered'].clear()

    async def __start_round(self, duration):
        """
        Message all participants the start of the current round, as well as the duration

        Because having somewhat synchronized timing is key to proper market operation, the start round message mostly
        contains useful time intervals. Additional info that are deemed useful can be included,
        such as grid prices that change with time.
        As always, it is advised to keep the message length minimal to maximize performance and to conserve bandwidth.

        Participants can take the times in this message and determine clock differences and communication delays.
        Will be necessary in real-time to ensure actions are received by the market before the start of the next round,
        as the market does not wait in real-time mode
        """
        start_time = self.__time()
        self.__reset_status()
        market_info = {
            str(self.__timing['current_round']): {
                'grid': {
                    'buy_price': self.__grid.buy_price(),
                    'sell_price': self.__grid.sell_price()
                }
            },
            str(self.__timing['next_settle']): {
                'grid': {
                    'buy_price': self.__grid.buy_price(),
                    'sell_price': self.__grid.sell_price()
                }
            },
        }

        start_msg = {
            'time': start_time,
            'duration': duration,
            'timezone': self.__timing['timezone'],
            'last_round': self.__timing['last_round'],
            'current_round': self.__timing['current_round'],
            'last_settle': self.__timing['last_settle'],
            'next_settle': self.__timing['next_settle'],
            'market_info': market_info,
        }
        # await self.__client.emit('start_round', start_msg)
        self.__client.publish('/'.join([self.market_id, 'start_round']), start_msg)


    async def submit_bid(self, message: dict):
        """Processes bids sent from the participants

        If action from participants are valid, then an entry will be made on the market for matching.
        In all cases, a confirmation message will be sent back to the sender indicating success or failure.
        The handling of the confirmation message is up to the participant.

        If the message and entry_type are valid, an open record will be made in the time delivery slot for source type.
        the record is a dictionary containing the following:

        - 'uuid'
        - 'participant_id'
        - 'session_id'
        - 'price'
        - 'time_submission'
        - 'quantity'
        - 'lock'

        Note: as of April 1, 2020, 'lock' is not being used in simulation mode.
        Deprecation in general is under consideration

        Parameters
        ----------
        message : dict
            Message should be a dictionary containing the following:
            - 'participant_id'
            - 'quantity' (quantity in Wh)
            - 'price' (price in $/kWh)
            - 'time_delivery'

        Returns
        -------
        confirmation
            returns the participant session id and confirmation message for SIO server callback

            - For all invalid entries, confirmation message is a dictionary with 'uuid' as the key and None as the value
            - For all valid entries, confirmation message be a dictionary containing the following:

                - 'uuid'
                - 'time_submission'
                - 'price'
                - 'quantity'
                - 'time_delivery'

        """

        # entry validity check step 1: quantity must be positive
        if message['quantity'] <= 0:
            # raise Exception('quantity must be a positive integer')
            return
            # return message['participant_id'], {'uuid': None}

        # if entry is valid, then update entry with market specific info
        # convert kwh price to token price

        entry = {
            'uuid': cuid().generate(),
            'participant_id': message['participant_id'],
            # 'session_id': message['session_id'],
            'price': message['price'],
            'time_submission': self.__time(),
            'quantity': message['quantity'],
            'lock': False
        }

        # create a new time slot container if the time slot doesn't exist
        time_delivery = tuple(message['time_delivery'])
        if time_delivery not in self.__open:
            self.__open[time_delivery] = {
                'bid': []
            }

        # if the time slot exists but no entry exist, create the entry container
        if 'bid' not in self.__open[time_delivery]:
            self.__open[time_delivery]['bid'] = []

        # add open entry
        self.__open[time_delivery]['bid'].append(entry)

        reply = {
            'uuid': entry['uuid'],
            'time_submission': entry['time_submission'],
            'price': entry['price'],
            'quantity': entry['quantity'],
            'time_delivery': time_delivery
        }

        self.__client.publish('/'.join([self.market_id, message['participant_id'], 'bid_success']), reply)
        # return message['participant_id'], reply

    async def submit_ask(self, message: dict):
        """Processes bids/asks sent from the participants

        If action from participants are valid, then an entry will be made on the market for matching.
        In all cases, a confirmation message will be sent back to the sender indicating success or failure.
        The handling of the confirmation message is up to the participant.

        If the message and entry_type are valid, an open record will be made in the time delivery slot for source type.
        the record is a dictionary containing the following:

        - 'uuid'
        - 'participant_id'
        - 'session_id'
        - 'source'
        - 'price'
        - 'time_submission'
        - 'quantity'
        - 'lock'

        Note: as of April 1, 2020, 'lock' is not being used in simulation mode.
        Deprecation in general is under consideration

        Parameters
        ----------
        message : dict
            Message should be a dictionary containing the following:

            - 'participant_id'
            - 'quantity' (quantity in Wh)
            - 'price' (price in $/kWh)
            - 'source' (such as 'solar', 'bess', 'wind', etc. Must be classifiable)
            - 'time_delivery'

        entry_type : str
            Must be either 'bid' or 'ask'

        Returns
        -------
        confirmation
            returns the participant session id and confirmation message for SIO server callback

            - For all invalid entries, confirmation message is a dictionary with 'uuid' as the key and None as the value
            - For all valid entries, confirmation message be a dictionary containing the following:

                - 'uuid'
                - 'time_submission'
                - 'source'
                - 'price'
                - 'quantity'
                - 'time_delivery'

        """

        # if entry_type not in {'bid', 'ask'}:
        #     # raise Exception('invalid action')
        #     return message['session_id'], {'uuid': None}

        # entry validity check step 1: quantity must be positive
        if message['quantity'] <= 0:
            # raise Exception('quantity must be a positive integer')
            # return message['session_id'], {'uuid': None}
            return

        # entry validity check step 2: source must be classifiable
        source_type = await self.__classify_source(message['source'])
        if not source_type:
            # raise Exception('quantity must be a positive integer')
            # return message['session_id'], {'uuid': None}
            return

        # if entry is valid, then update entry with market specific info
        # convert kwh price to token price

        entry = {
            'uuid': cuid().generate(),
            'participant_id': message['participant_id'],
            # 'session_id': message['session_id'],
            'source': message['source'],
            'price': message['price'],
            'time_submission': self.__time(),
            'quantity': message['quantity'],
            'lock': False
        }

        # create a new time slot container if the time slot doesn't exist
        time_delivery = tuple(message['time_delivery'])
        if time_delivery not in self.__open:
            self.__open[time_delivery] = {
                'ask': []
            }

        # if the time slot exists but no entry exist, create the entry container
        if 'ask' not in self.__open[time_delivery]:
            self.__open[time_delivery]['ask'] = []

        # add open entry
        self.__open[time_delivery]['ask'].append(entry)

        reply = {
            'uuid': entry['uuid'],
            'time_submission': entry['time_submission'],
            'source': entry['source'],
            'price': entry['price'],
            'quantity': entry['quantity'],
            'time_delivery': time_delivery
        }
        self.__client.publish('/'.join([self.market_id, message['participant_id'], 'ask_success']), reply)

        # return message['session_id'], reply

    async def __match(self, time_delivery):
        """Matches bids with asks for a single source type in a time slot

        THe matching and settlement process closely resemble double auctions.
        For all bids/asks for a source in the delivery time slots, highest bids are matched with lowest asks
        and settled pairwise. Quantities can be partially settled. Unsettled quantities are discarded. Participants are only obligated to buy/sell quantities settled for the delivery period.

        Parameters
        ----------
        time_delivery : tuple
            Tuple containing the start and end timestamps in UNIX timestamp format indicating the interval for energy to be delivered.

        Notes
        -----
        Presently, the settlement price is hard-coded as the average price of the bid/ask pair. In the near future, dedicated, more sophisticated functions for determining settlement price will be implemented

        """

        if time_delivery not in self.__open:
            return

        if 'ask' not in self.__open[time_delivery]:
            return

        if 'bid' not in self.__open[time_delivery]:
            return

        # remove zero-quantity bid and ask entries
        # sort bids by decreasing price and asks by increasing price
        # def filter_bids_asks():
        self.__open[time_delivery]['ask'][:] = \
            sorted([ask for ask in self.__open[time_delivery]['ask'] if ask['quantity'] > 0],
                   key=itemgetter('price'), reverse=False)
        self.__open[time_delivery]['bid'][:] = \
            sorted([bid for bid in self.__open[time_delivery]['bid'] if bid['quantity'] > 0],
                   key=itemgetter('price'), reverse=True)

        # await asyncio.get_event_loop().run_in_executor(filter_bids_asks)

        bids = self.__open[time_delivery]['bid']
        asks = self.__open[time_delivery]['ask']

        # await asyncio.get_event_loop().run_in_executor(send_request)

        for bid, ask, in itertools.product(bids, asks):
            if ask['price'] > bid['price']:
                continue

            if bid['participant_id'] == ask['participant_id']:
                continue

            # if bid['source'] != ask['source']:
            #     continue

            if bid['lock'] or ask['lock']:
                continue

            if bid['quantity'] <= 0 or ask['quantity'] <= 0:
                continue

            if bid['participant_id'] not in self.__participants:
                bid['lock'] = True
                continue

            if ask['participant_id'] not in self.__participants:
                ask['lock'] = True
                continue

            # Settle highest price bids with lowest price asks
            await self.__settle(bid, ask, time_delivery)

    async def __settle(self, bid: dict, ask: dict, time_delivery: tuple, settlement_method=None, locking=False):
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

        # only proceed to settle if settlement quantity is positive
        quantity = min(bid['quantity'], ask['quantity'])
        if quantity <= 0:
            return

        if locking:
            # lock the bid and ask until confirmations are received
            ask['lock'] = True
            bid['lock'] = True

        commit_id = cuid().generate()
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
            'lock': locking
        }

        # if buyer == 'grid' or seller == 'grid':
        # if buy_price is not None and sell_price is not None:
        #     return
        buyer_message = {
            'commit_id': commit_id,
            'bid_id': bid['uuid'],
            'source': ask['source'],
            'quantity': quantity,
            'buy_price': settlement_price_buy,
            'time_delivery': time_delivery
        }

        seller_message = {
            'commit_id': commit_id,
            'ask_id': ask['uuid'],
            'source': ask['source'],
            'quantity': quantity,
            'sell_price': settlement_price_sell,
            'time_delivery': time_delivery
        }
        # self.__client.publish('/'.join([self.market_id, bid['participant_id'], 'send_settlement']), buyer_message)
        # self.__client.publish('/'.join([self.market_id, ask['participant_id'], 'send_settlement']), seller_message)
        self.__client.publish('/'.join([self.market_id, bid['participant_id'], 'settled']), buyer_message)
        self.__client.publish('/'.join([self.market_id, ask['participant_id'], 'settled']), seller_message)

        bid['quantity'] = max(0, bid['quantity'] - self.__settled[time_delivery][commit_id]['record']['quantity'])
        ask['quantity'] = max(0, ask['quantity'] - self.__settled[time_delivery][commit_id]['record']['quantity'])
        self.__status['round_settled'].append(commit_id)

    # after settlement confirmation, update bid and ask quantities
    async def settlement_delivered(self, commit_id):
        # self.__status['round_settle_delivered'].append(commit_id)
        if commit_id not in self.__status['round_settle_delivered']:
            self.__status['round_settle_delivered'][commit_id] = 1
        else:
            self.__status['round_settle_delivered'][commit_id] += 1

    # async def __settle_confirm_lock(self, message):
    #     """Callback for settle in locking mode
    #
    #     """
    #
    #     time_delivery = tuple(message['time_delivery'])
    #     if time_delivery not in self.__settled:
    #         return
    #
    #     commit_id = message['commit_id']
    #     ask = self.__settled[time_delivery][commit_id]['ask']
    #     bid = self.__settled[time_delivery][commit_id]['bid']
    #
    #     ask['lock'] = not message['seller']
    #     bid['lock'] = not message['buyer']
    #
    #     if not ask['lock'] and not bid['lock']:
    #         self.__settled[time_delivery][commit_id]['lock'] = False
    #         bid['quantity'] = max(0, bid['quantity'] - self.__settled[time_delivery][commit_id]['record']['quantity'])
    #         ask['quantity'] = max(0, ask['quantity'] - self.__settled[time_delivery][commit_id]['record']['quantity'])

    async def meter_data(self, message):
        """Update meter data from participant

        Meter data should be received from participants at the end of the each round for delivery.
        """

        # meter = {
        #     'time_interval': (),
        #     'generation': {
        #         'solar': 0,
        #         'bess': 0
        #     },
        #     'consumption': {
        #         'bess': {
        #             'solar': 0,
        #         },
        #         'other': {
        #             'solar': 0,
        #             'bess': 0,
        #             'other': 0
        #         }
        #     }
        # }

        # TODO: add data validation later
        # print(message)
        participant_id = message['participant_id']
        time_delivery = tuple(message['meter']['time_interval'])
        self.__participants[participant_id]['meter'][time_delivery] = message['meter']
        self.__status['round_metered'] += 1

    async def __process_settlements(self, time_delivery, source_type):
        physical_tranactions = []
        financial_transactions = []
        settlements = self.__settled[time_delivery]
        for buyer in self.__participants:
            for seller in self.__participants:
                if buyer == seller:
                    continue
                # make sure the buyer and seller are online
                if not self.__participants[buyer]['online']:
                    continue
                if not self.__participants[seller]['online']:
                    continue

                # Extract settlements involving buyer and seller (that are not locked)
                relevant_settlements = {k: v for (k, v) in settlements.items() if
                                        settlements[k]['lock'] is False and
                                        settlements[k]['buyer_id'] == buyer and
                                        settlements[k]['seller_id'] == seller}

                if relevant_settlements:
                    for commit_id in relevant_settlements.keys():
                        energy_source = self.__settled[time_delivery][commit_id]['source']
                        energy_type = await self.__classify_source(energy_source)
                        if energy_type != source_type:
                            continue

                        settled_quantity = self.__settled[time_delivery][commit_id]['record']['quantity']
                        if not settled_quantity:
                            continue
                        residual_generation = self.__participants[seller]['meter'][time_delivery]['generation'][
                            energy_source]
                        residual_consumption = \
                        self.__participants[buyer]['meter'][time_delivery]['consumption']['other']['external']

                        # check to see if physical generation is less than settled quantity
                        # extra_purchase = 0
                        deficit_generation = max(0, settled_quantity - residual_generation)
                        # Add on the amount that needed to be bought from the grid?
                        # self.__participants[buyer]['meter']['consumption']['other']['external'] += deficit_generation
                        # if not deficit_generation:
                        # check if settled quantity is greater than residual consumption
                        # if settled amount is greater than residual generation, then figure out
                        # the financial compensation.
                        extra_purchase = max(0, settled_quantity - residual_consumption)
                        # print(settled_quantity, energy_source, residual_generation, residual_consumption, extra_purchase, deficit_generation)
                        pt, ft = await self.__transfer_energy(time_delivery, commit_id, extra_purchase,
                                                              deficit_generation)
                        physical_tranactions.extend(pt)
                        financial_transactions.extend(ft)
        return physical_tranactions, financial_transactions

    # async def __process_self_consumption(self, participant_id):

    async def __scrub_financial_transaction(self, transactions):
        scrubbed_transactions = {}
        for transaction in transactions:
            if transaction['seller_id'] not in scrubbed_transactions:
                scrubbed_transactions[transaction['seller_id']] = {
                    'buy': [],
                    'sell': []
                }
            if transaction['buyer_id'] not in scrubbed_transactions:
                scrubbed_transactions[transaction['buyer_id']] = {
                    'buy': [],
                    'sell': []
                }
            scrubbed_transaction = {
                'quantity': transaction['quantity'],
                'energy_source': transaction['energy_source'],
                'settlement_price_sell': transaction['settlement_price_sell'],
                'settlement_price_buy': transaction['settlement_price_buy'],
                'time_creation': transaction['time_creation'],
                'time_purchase': transaction['time_purchase']
            }
            scrubbed_transactions[transaction['buyer_id']]['buy'].append(scrubbed_transaction)
            scrubbed_transactions[transaction['seller_id']]['sell'].append(scrubbed_transaction)
        return scrubbed_transactions

    async def __process_energy_exchange(self, time_delivery):
        """The main function for finalizing energy exchange using settlements and meter data.

        Energy exchange takes the following steps in order of priority:

        1. Exchange settled energy
        2. Process self-consumption
        3. Process residual energy

        Aside from perfect settlements (i.e, ), which are not expected in realistic scen

        There are five possible scenarios for each settlement:

        1. Seller's residual generation is the exact amount as settled
        2. Seller's residual generation is less than settled
        3. Seller's residual generation is more than settled
        4. Buyer's residual consumption is the exact amount as settled
        5. Buyer's residual consumption is less than settled
        6. Buyer's residual consumption is more than settled

        Aside from 1 and 4, all other scenarios require additional handling.

        - Scenario 2: The seller must either pay for the shortage from the grid, or compensate by injecting the shortage from their BESS. BESS compensation must be done prior to sending meter data.
        - Scenario 3: The residual are sold to the grid at grid prices
        - Scenario 5: The buyer must pay the seller the full amount of the settlement. The residual generation cannot be sold to the grid again, as that would be double compensation.
        - Scenario 6: The buyer must buy the residual consumption from the grid at grid prices.

        On top of properly balancing the market, these schemes should also provide sufficient punishment that drive the agents to make more optimal decisions.

        """
        # print('-----')
        # STEP 1
        # process auction deliveries
        transactions = []
        financial_transactions = []
        # Step 1: exchange settled
        if time_delivery in self.__settled:
            for source_type in {'dispatch', 'non_dispatch'}:
                # important: dispatch must be first!!!
                pt, ft = await self.__process_settlements(time_delivery, source_type)
                transactions.extend(pt + ft)
                financial_transactions.extend(ft)

        scrubbed_financial_transactions = await self.__scrub_financial_transaction(financial_transactions)

        # Steps 2 & 3
        # process self-consumption
        # process residual energy
        for participant_id in self.__participants:
            if not self.__participants[participant_id]['meter']:
                continue

            if time_delivery not in self.__participants[participant_id]['meter']:
                print(participant_id, 'not metered')
                continue

            # self consumption
            for load in self.__participants[participant_id]['meter'][time_delivery]['consumption']:
                for source in self.__participants[participant_id]['meter'][time_delivery]['consumption'][load]:
                    if source in self.__participants[participant_id]['meter'][time_delivery]['generation']:
                        # assuming everything is perfectly sub metered
                        quantity = self.__participants[participant_id]['meter'][time_delivery]['consumption'][load][
                            source]

                        if quantity > 0:
                            transaction_record = {
                                'quantity': quantity,
                                'seller_id': participant_id,
                                'buyer_id': participant_id,
                                'energy_source': source,
                                'settlement_price_sell': 0,
                                'settlement_price_buy': 0,
                                'time_creation': time_delivery[0],
                                'time_purchase': time_delivery[1],
                                'time_consumption': time_delivery[1]
                            }
                            transactions.append(transaction_record.copy())
                            self.__participants[participant_id]['meter'][time_delivery]['consumption'][load][
                                source] -= quantity

            extra_transactions = {
                'participant': participant_id,
                'time_delivery': time_delivery,
                'grid': {
                    'buy': [],
                    'sell': []
                }
            }
            # sell residual generation(s) to the grid
            for source in self.__participants[participant_id]['meter'][time_delivery]['generation']:
                residual_generation = self.__participants[participant_id]['meter'][time_delivery]['generation'][source]
                if residual_generation > 0:
                    transaction_record = {
                        'quantity': residual_generation,
                        'seller_id': participant_id,
                        'buyer_id': self.__grid.id,
                        'energy_source': source,
                        'settlement_price_sell': self.__grid.sell_price(),
                        'settlement_price_buy': self.__grid.sell_price(),
                        'time_creation': time_delivery[0],
                        'time_purchase': time_delivery[1],
                        'time_consumption': time_delivery[1]
                    }
                    transactions.append(transaction_record.copy())
                    self.__participants[participant_id]['meter'][time_delivery]['generation'][
                        source] -= residual_generation
                    extra_transactions['grid']['sell'].append(transaction_record.copy())
            # buy residual consumption (other) from grid
            residual_consumption = self.__participants[participant_id]['meter'][time_delivery]['consumption']['other'][
                'external']
            if residual_consumption > 0:
                transaction_record = {
                    'quantity': residual_consumption,
                    'seller_id': self.__grid.id,
                    'buyer_id': participant_id,
                    'energy_source': 'grid',
                    'settlement_price_sell': self.__grid.buy_price(),
                    'settlement_price_buy': self.__grid.buy_price(),
                    'time_creation': time_delivery[0],
                    'time_purchase': time_delivery[1],
                    'time_consumption': time_delivery[1]
                }
                transactions.append(transaction_record.copy())
                self.__participants[participant_id]['meter'][time_delivery]['consumption']['other'][
                    'external'] -= residual_consumption
                extra_transactions['grid']['buy'].append(transaction_record.copy())

            if participant_id in scrubbed_financial_transactions:
                extra_transactions['financial'] = scrubbed_financial_transactions[participant_id]

            # await self.__client.emit(event='return_extra_transactions',
            #                          data=extra_transactions)
            topic = '/'.join([self.market_id, participant_id, 'return_extra_transactions'])
            self.__client.publish(topic, extra_transactions)


        if self.save_transactions:
            self.__transactions.extend(transactions)
            await self.record_transactions(10000)

    async def __transfer_energy(self, time_delivery, commit_id, extra_purchase=0, deficit_generation=0):
        # pt, ft = await self.__transfer_energy(time_delivery, commit_id, extra_purchase, deficit_generation)
        """This function makes the energy transaction records for each settlement

        """

        physical_transactions = []
        financial_transactions = []
        seller_id = self.__settled[time_delivery][commit_id]['seller_id']
        buyer_id = self.__settled[time_delivery][commit_id]['buyer_id']
        energy_source = self.__settled[time_delivery][commit_id]['source']
        physical_qty = 0
        settlement = self.__settled[time_delivery][commit_id]['record']
        # For extra consumption by buyer greater than settled amount:
        physical_record = settlement.copy()

        # extra purchase by buyer
        # buyer settled for more than consumed
        if extra_purchase:
            print('-extra---------')
            print(buyer_id, extra_purchase)
            print(settlement)
            print(self.__participants[buyer_id]['meter'][time_delivery])

        # extra_purchase and deficit_generation SHOULD be mutually exclusive

        if not extra_purchase and not deficit_generation:
            physical_record.update({
                'time_creation': time_delivery[0],
                'time_consumption': time_delivery[1],
            })
            physical_qty = physical_record['quantity']
            self.__participants[seller_id]['meter'][time_delivery]['generation'][energy_source] -= physical_qty
            self.__participants[buyer_id]['meter'][time_delivery]['consumption']['other']['external'] -= physical_qty
            physical_transactions.append(physical_record)
        # settled for more than consumed
        elif extra_purchase:
            physical_record.update({
                'quantity': physical_record['quantity'] - extra_purchase,
                'time_creation': time_delivery[0],
                'time_consumption': time_delivery[1],
            })
            financial_record = settlement.copy()
            financial_record.update({
                'quantity': extra_purchase,
                'time_creation': time_delivery[0]
            })
            financial_transactions.append(financial_record)

            if physical_record['quantity']:
                physical_qty = physical_record['quantity']
                self.__participants[seller_id]['meter'][time_delivery]['generation'][energy_source] -= physical_qty
                self.__participants[buyer_id]['meter'][time_delivery]['consumption']['other'][
                    'external'] -= physical_qty
                physical_transactions.append(physical_record)

        elif deficit_generation:
            # print('-=-------------=-')
            # print(settlement)
            # print(short)
            # print(self.__participants[seller_id]['meter']['generation']['bess'])

            # seller makes up for less than promised by
            # first, compensate from battery (if extra discharge). These are physical
            # second, financially compensate by buying energy from grid for buyer. These are financial.

            # battery can only compensate for non-dispatch settlements for now
            source_type = await self.__classify_source(settlement['energy_source'])
            if source_type == 'non_dispatch':
                residual_bess = self.__participants[seller_id]['meter'][time_delivery]['generation']['bess']
                bess_compensation = min(deficit_generation, residual_bess)
                # print(deficit_generation,
                #       bess_compensation,
                #       self.__participants[seller_id]['meter'][time_delivery]['generation']['bess'],
                #       self.__participants[seller_id]['meter'][time_delivery]['generation']['solar'],
                #       physical_qty)

                if bess_compensation > 0:
                    compensation_record = {
                        'quantity': bess_compensation,
                        'seller_id': settlement['seller_id'],
                        'buyer_id': settlement['buyer_id'],
                        'energy_source': 'bess',
                        'settlement_price_sell': settlement['settlement_price_sell'],
                        'settlement_price_buy': settlement['settlement_price_buy'],
                        'time_creation': time_delivery[0],
                        'time_purchase': settlement['time_purchase'],
                        'time_consumption': time_delivery[1]
                    }
                    self.__participants[seller_id]['meter'][time_delivery]['generation']['bess'] -= bess_compensation
                    self.__participants[buyer_id]['meter'][time_delivery]['consumption']['other'][
                        'external'] -= bess_compensation
                    deficit_generation -= bess_compensation
                    physical_transactions.append(compensation_record)

                    # print(deficit_generation,
                    #       bess_compensation,
                    #       self.__participants[seller_id]['meter'][time_delivery]['generation']['bess'],
                    #       self.__participants[seller_id]['meter'][time_delivery]['generation']['solar'],
                    #       physical_qty)

            # if deficit_generation:
            #     # print(extra_purchase, deficit_generation)
            #     print('-short---------')
            #     # print(buyer_id, extra_purchase)
            #     print(seller_id, deficit_generation)
            #     print(settlement)
            #     print(self.__participants[seller_id]['meter'][time_delivery])

            if deficit_generation > 0:
                financial_record = {
                    'quantity': deficit_generation,
                    'seller_id': seller_id,
                    'buyer_id': buyer_id,
                    'energy_source': 'grid',
                    'settlement_price_sell': 0,
                    'settlement_price_buy': -self.__grid.buy_price(),  # seller pays buyer
                    'time_creation': time_delivery[0],
                    'time_purchase': time_delivery[1]
                }
                financial_transactions.append(financial_record)

        await self.__complete_settlement(time_delivery, commit_id)
        return physical_transactions, financial_transactions

    # async def __complete_settlement_cb(self, time_delivery, commit_id):
    #     if not commit_id:
    #         return
    #     time_delivery = tuple(time_delivery)
    #     if time_delivery not in self.__settled:
    #         return
    #     if commit_id not in self.__settled[time_delivery]:
    #         return
    #     del self.__settled[time_delivery][commit_id]

    # mark completion of successful settlements
    async def __complete_settlement(self, time_delivery, commit_id):
        # message = {
        #     'time_delivery': time_delivery,
        #     'commit_id': commit_id,
        #     'seller_id': self.__settled[time_delivery][commit_id]['seller_id'],
        #     'buyer_id': self.__settled[time_delivery][commit_id]['buyer_id']
        # }
        # await self.__client.emit('settlement_complete', message, namespace='/market', callback=self.__complete_settlement_cb)
        # await self.__client.emit('settlement_complete', message, namespace='/market')
        del self.__settled[time_delivery][commit_id]

    @tenacity.retry(wait=tenacity.wait_fixed(5))
    async def __ensure_transactions_complete(self):
        table_len = db_utils.get_table_len(self.__db['path'], self.__db['table'])
        if table_len < self.transactions_count:
            raise Exception
        return True

    async def record_transactions(self, buf_len=0, delay=True, check_table_len=False):
        """This function records the transaction records into the ledger

        """

        if check_table_len:
            table_len = db_utils.get_table_len(self.__db['path'], self.__db['table'])
            if table_len < self.transactions_count:
                return False

        if delay and buf_len:
            delay = buf_len / 100
            ts = datetime.datetime.now().timestamp()
            if ts - self.__transaction_last_record_time < delay:
                return False

        transactions_len = len(self.__transactions)
        if transactions_len < buf_len:
            return False

        transactions = self.__transactions[:transactions_len]
        asyncio.create_task(db_utils.dump_data(transactions, self.__db['path'], self.__db['table']))

        self.__transaction_last_record_time = datetime.datetime.now().timestamp()
        del self.__transactions[:transactions_len]
        self.transactions_count += transactions_len
        return True

    async def __clean_market(self, time_delivery):
        # clean buffer from 2 rounds before the current round
        # ensure this will not interfere with settlement callbacks
        duration = self.__timing['duration']
        time_clean = (time_delivery[0] - duration, time_delivery[1] - duration)
        self.__open.pop(time_clean, None)
        self.__settled.pop(time_clean, None)
        for participant in self.__participants:
            self.__participants[participant]['meter'].pop(time_delivery, None)

    async def __update_time(self, time):
        self.__server_ts = time['time']
        duration = time['duration']
        start_time = time['time']
        end_time = start_time + duration
        self.__timing.update({
            'timezone': self.__timing['timezone'],
            'duration': duration,
            'last_round': self.__timing['current_round'],
            'current_round': (start_time, end_time),
            'last_settle': (start_time + duration * (self.__timing['close_steps'] - 1),
                            start_time + duration * self.__timing['close_steps']),
            'next_settle': (start_time + duration * self.__timing['close_steps'],
                            start_time + duration * (self.__timing['close_steps'] + 1))
            # 'next_settle': (1433152800, 1433149200)
        })
        # print(self.__timing)

    # Make sure time interval provided is valid
    async def __time_interval_is_valid(self, time_interval: tuple):
        duration = self.__timing['duration']
        if (time_interval[1] - time_interval[0]) % duration != 0:
            # make sure duration is a multiple of round duration
            return False
        if time_interval[0] % duration != 0:
            return False
        if time_interval[1] % duration != 0:
            return False
        return True

    async def __match_all(self, time_delivery):
        await self.__match(time_delivery)
        self.__status['round_matched'] = True

    # should be for simulation mode only
    @tenacity.retry(wait=tenacity.wait_fixed(0.01))
    async def __ensure_round_complete(self):
        # print(self.__status)
        if self.__status['round_metered'] < self.__status['active_participants']:
            raise Exception

        if not self.__status['round_matched']:
            raise Exception

        keys = [k for k, v in self.__status['round_settle_delivered'].items() if v == 2]
        if set(keys) != set(self.__status['round_settled']):
        # if set(self.__status['round_settle_delivered']) != set(self.__status['round_settled']):
            raise Exception

    # Finish all processes and remove all unnecessary/ remaining records in preparation for a new time step, begin processes for next step
    async def step(self, timeout=60, sim_params=None):
        # timing for simulation mode and real-time mode a slightly different due to one with an explicit end condition. RT mode sequence is not too relevant at the moment will be added later.
        if self.__timing['mode'] == 'sim':
            await self.__update_time(sim_params)
            if not self.__timing['current_round'][0] % 3600:
                self.__grid.update_price(self.__timing['current_round'][0], self.__timing['timezone'])
            await self.__start_round(duration=timeout)
            await self.__match_all(self.__timing['last_settle'])
            await self.__ensure_round_complete()
            # print(self.__status)
            # print('round complete?')
            await self.__process_energy_exchange(self.__timing['current_round'])
            await self.__clean_market(self.__timing['last_round'])
            # await self.__client.emit('end_round', data='')

            self.__client.publish('/'.join([self.market_id, 'simulation', 'end_round']), '')

    async def loop(self):
        # change loop depending on sim mode or RT mode
        while self.run:
            if self.server_online and self.__timing['mode'] == 'rt':
                await self.step(60)
            # continue
            await asyncio.sleep(1)
        else:
            await asyncio.sleep(5)
            await self.__client.disconnect()
            os.kill(os.getpid(), signal.SIGINT)
            raise SystemExit

        # raise SystemExit

    async def reset_market(self):
        self.__db.clear()
        self.transactions_count = 0
        self.__open.clear()
        self.__settled.clear()
        for participant in self.__participants:
            self.__participants[participant]['meter'].clear()

    async def end_sim_generation(self):
        await self.record_transactions(delay=False)
        await self.__ensure_transactions_complete()
        await self.reset_market()
        # await self.__client.emit('market_ready')
        self.__client.publish('/'.join([self.market_id, 'simulation', 'market_ready']), '')

    async def market_is_online(self):
        self.__client.publish('/'.join([self.market_id, 'simulation', 'market_online']), '')

    async def kill(self):
        await asyncio.sleep(5)
        await self.__client.disconnect()
        # print('attempting to end')
        os.kill(os.getpid(), signal.SIGINT)
        raise SystemExit