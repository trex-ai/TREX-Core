"""This implements a tabular epsilon greedy Q-learning for a MicroTE Bandit problem

"""
import asyncio
import importlib
import random
import numpy as np
from _agent._utils.metrics import Metrics
from _utils import utils, db_utils

import sqlalchemy
from sqlalchemy import MetaData, Column
import dataset
import ast

class Trader:
    """This trader uses SMA crossover to make trading decisions in the context of MicroFE

    The trader tries to learn the right prices for each minute of the day. This is done by initializing two prices tables, one for bid prices and one for ask prices. Each table is 1440 elements long. The tables are initialized by randomizing prices of each minute within a price range. A 15 minute window is used for initialization, which means that only 96 initial prices are generated. This is meant to decrease initial noise. Successful trades will nudge bid and ask prices to the point of most profit and least cost.
    """
    def __init__(self, bid_price, ask_price, **kwargs):
        # Some utility parameters
        self.__participant = kwargs['trader_fns']
        self.status = {
            'weights_loading': False
        }

        self.min_price = min(bid_price, ask_price)
        self.max_price = max(bid_price, ask_price)

        # Initialize the agent learning parameters
        self.bid_prices = dict()
        self.ask_prices = dict()

        self.bid_price = dict()
        self.ask_price = dict()
        # self.bid_price = utils.secure_random.choice(np.linspace(bid_price, ask_price, 11))
        # self.ask_price = utils.secure_random.choice(np.linspace(bid_price, ask_price, 11))

        if 'storage' in self.__participant:
            self.storage_prices = dict()
            self.storage_qtys = dict()

            self.storage_price = dict()
            self.storage_qty = dict()
            # self.storage_price = utils.secure_random.choice(np.linspace(-bid_price, bid_price, 21))
            # self.storage_qty = utils.secure_random.choice(np.linspace(-10, 10, 21))
        # Initialize learning parameters
        self.learning = kwargs['learning'] if 'learning' in kwargs else False
        reward_function = kwargs['reward_function'] if 'reward_function' in kwargs else None
        if reward_function:
            self._rewards = importlib.import_module('_agent.rewards.' + reward_function).Reward(
                self.__participant['timing'],
                self.__participant['ledger'],
                self.__participant['market_info'])

        self.learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs else 0.1
        self.discount_factor = kwargs['discount_factor'] if 'discount_factor' in kwargs else 0.99
        self.exploration_factor = kwargs['exploration_factor'] if 'exploration_factor' in kwargs else 0.1

        # Initialize metrics tracking
        self.track_metrics = kwargs['track_metrics'] if 'track_metrics' in kwargs else False
        self.metrics = Metrics(self.__participant['id'], track=self.track_metrics)
        if self.track_metrics:
            self.__init_metrics()

    def __init_metrics(self):
        import sqlalchemy
        '''
        Initializes metrics to record into database
        '''
        self.metrics.add('timestamp', sqlalchemy.Integer)
        self.metrics.add('actions_dict', sqlalchemy.JSON)
        self.metrics.add('rewards', sqlalchemy.Float)
        self.metrics.add('next_settle_load', sqlalchemy.Integer)
        self.metrics.add('next_settle_generation', sqlalchemy.Integer)
        if 'storage' in self.__participant:
            self.metrics.add('storage_soc', sqlalchemy.Float)

    def __select_price(self, bid_price, ask_price):
        bid = int(bid_price * 10000)
        ask = int(ask_price * 10000)
        price = utils.secure_random.randint(*sorted([bid, ask]))/10000
        return price

    def __generate_price_table(self, bid_price, ask_price, number):
        price_table = {price: 0 for price in np.linspace(bid_price, ask_price, number)}
        return price_table

    def anneal(self, parameter:str, adjustment, mode:str='multiply'):
        if not hasattr(self, parameter):
            return False

        if mode not in ('subtract', 'multiply'):
            return False

        param_value = getattr(self, parameter)
        if mode == 'subtract':
            param_value = max(0, param_value - adjustment)
        elif mode == 'multiply':
            param_value *= adjustment
        setattr(self, parameter, param_value)

    def check_state_exists(self, q_table, major_key, minor_key):
        if major_key not in q_table:
            # return False
            q_table[major_key] = {minor_key: 0}
        if minor_key not in q_table[major_key]:
            # return False
            q_table[major_key][minor_key] = 0

    # Core Functions, learn and act, called from outside
    async def learn(self, **kwargs):
        if not self.learning:
            return

        reward = await self._rewards.calculate()
        await self.metrics.track('rewards', reward)

        if reward is None:
            return

        last_deliver = self.__participant['timing']['last_deliver']

        bid_price = self.bid_price.pop(last_deliver, None)
        if bid_price:
            self.check_state_exists(self.bid_prices, last_deliver, bid_price)
            q_bid = self.bid_prices[last_deliver][bid_price]
            q_max_bid = max(self.bid_prices[last_deliver].values())
            q_bid_new = q_bid + self.learning_rate * (reward + self.discount_factor * q_max_bid - q_bid)
            self.bid_prices[last_deliver][bid_price] = q_bid_new

        ask_price = self.ask_price.pop(last_deliver, None)
        if ask_price:
            self.check_state_exists(self.ask_prices, last_deliver, ask_price)
            q_ask = self.ask_prices[last_deliver][ask_price]
            q_max_ask = max(self.ask_prices[last_deliver].values())
            q_ask_new = q_ask + self.learning_rate * (reward + self.discount_factor * q_max_ask - q_ask)
            self.ask_prices[last_deliver][ask_price] = q_ask_new

        if 'storage' in self.__participant:
            storage_price = self.storage_price.pop(last_deliver, None)
            if storage_price is not None:
                self.check_state_exists(self.storage_prices, last_deliver, storage_price)
                q_storage_price = self.storage_prices[last_deliver][storage_price]
                q_max_storage_price = max(self.storage_prices[last_deliver].values())
                q_storage_price_new = q_storage_price + self.learning_rate * \
                                      (reward + self.discount_factor * q_max_storage_price - q_storage_price)
                self.storage_prices[last_deliver][storage_price] = q_storage_price_new

            # if last_deliver not in self.storage_qtys:
            #     self.storage_qtys[last_deliver] = {self.storage_qty: 0}
            # elif self.storage_price not in self.storage_qtys[last_deliver]:
            #     self.storage_qtys[last_deliver][self.storage_qty] = 0
            storage_qty = self.storage_qty.pop(last_deliver, None)
            if storage_qty is not None:
                self.check_state_exists(self.storage_qtys, last_deliver, storage_qty)
                q_storage_qty = self.storage_qtys[last_deliver][storage_qty]
                q_max_storage_qty = max(self.storage_qtys[last_deliver].values())
                q_storage_qty_new = q_storage_qty + self.learning_rate * \
                                      (reward + self.discount_factor * q_max_storage_qty - q_storage_qty)
                self.storage_qtys[last_deliver][storage_qty] = q_storage_qty_new

    async def act(self, **kwargs):
        actions = {}
        next_settle = self.__participant['timing']['next_settle']

        epsilon = self.exploration_factor if self.learning else -1
        explore = utils.secure_random.random() <= epsilon or \
                  next_settle not in self.bid_prices or \
                  next_settle not in self.ask_prices

        # check if state exists. If not then add state and randomly choose action
        if explore:
            self.bid_price[next_settle] = utils.secure_random.choice(np.linspace(self.min_price, self.max_price, 11))
            self.ask_price[next_settle] = utils.secure_random.choice(np.linspace(self.min_price, self.max_price, 11))
        else:
            q_max_bid = max(self.bid_prices[next_settle].values())
            q_max_ask = max(self.ask_prices[next_settle].values())

            bid_argmaxes = [price for price, q in self.bid_prices[next_settle].items() if q == q_max_bid]
            ask_argmaxes = [price for price, q in self.ask_prices[next_settle].items() if q == q_max_ask]

            self.bid_price[next_settle] = utils.secure_random.choice(bid_argmaxes)
            self.ask_price[next_settle] = utils.secure_random.choice(ask_argmaxes)

        if 'storage' in self.__participant:
            # storage qty corresponds to battery scheduler. i.e.
            # +ve = charge battery = add load
            # -ve = discharge battery = add generation

            if explore or next_settle not in self.storage_prices or next_settle not in self.storage_qtys:
                self.storage_price[next_settle] = utils.secure_random.choice(np.linspace(self.min_price, self.max_price, 11))
                self.storage_qty[next_settle] = utils.secure_random.choice(np.linspace(-20, 20, 11))
            else:
                q_max_storage_price = max(self.storage_prices[next_settle].values())
                q_max_storage_qty = max(self.storage_qtys[next_settle].values())

                storage_price_argmaxes = [price for price, q in self.storage_prices[next_settle].items() if q == q_max_storage_price]
                storage_qty_argmaxes = [price for price, q in self.storage_qtys[next_settle].items() if q == q_max_storage_qty]

                self.storage_price[next_settle] = utils.secure_random.choice(storage_price_argmaxes)
                self.storage_qty[next_settle] = utils.secure_random.choice(storage_qty_argmaxes)

        next_generation, next_load = await self.__participant['read_profile'](next_settle)
        storage_delta = self.storage_qty[next_settle] if 'storage' in self.__participant else 0
        next_residual_load = next_load - next_generation + storage_delta
        next_residual_gen = -next_residual_load

        if next_residual_load > 0:
            actions['bids'] = {
                str(next_settle): {
                    'quantity': next_residual_load,
                    'price': self.bid_price[next_settle]
                }
            }

        if next_residual_gen > 0:
            actions['asks'] = {
                'solar': {
                    str(next_settle): {
                        'quantity': next_residual_gen,
                        'price': self.ask_price[next_settle]
                    }
                }
            }

        # make battery purely arbitrage for now:
        if 'storage' in self.__participant:
            if self.storage_qty[next_settle] > 0:
                actions['bids'] = {
                    str(next_settle): {
                        'quantity': self.storage_qty[next_settle],
                        'price': self.storage_price[next_settle]
                    }
                }
            elif self.storage_qty[next_settle] < 0:
                actions['asks']['bess'] = {
                    str(next_settle): {
                        'quantity': self.storage_qty[next_settle],
                        'price': self.storage_price[next_settle]
                    }
                }
            actions['bess'] = {
                str(next_settle): self.storage_qty[next_settle]
            }

            # if storage_delta < 0:
            #     if 'asks' not in actions:
            #         actions['asks'] = dict()
            #
            #     actions['asks']['bess'] = {
            #         str(next_settle): {
            #             'quantity': -storage_delta,
            #             'price': self.storage_price[next_settle]
            #         }
            #     }

        if self.track_metrics:
            await asyncio.gather(
                self.metrics.track('timestamp', self.__participant['timing']['current_round'][1]),
                self.metrics.track('actions_dict', actions),
                self.metrics.track('next_settle_load', next_load),
                self.metrics.track('next_settle_generation', next_generation))
        return actions

    async def save_weights(self, **kwargs):
        '''
        Save the price tables at the end of the episode into database
        '''

        table_name = '_'.join((str(kwargs['generation']), kwargs['market_id'], 'weights', self.__participant['id']))
        table = self.__create_weights_table(table_name)
        await db_utils.create_table(db_string=kwargs['db_path'],
                                    table_type='custom',
                                    custom_table=table)
        weights = [{
            'generation': kwargs['generation'],
            'bid_prices': str(self.bid_prices),
            'ask_prices': str(self.ask_prices)
        }]
        if 'storage' in self.__participant:
            weights[0].update({
                'storage_prices': str(self.storage_prices),
                'storage_qtys': str(self.storage_qtys)
            })

        await db_utils.dump_data(weights, kwargs['db_path'], table)

    def __create_weights_table(self, table_name):
        columns = [
            Column('generation', sqlalchemy.Integer, primary_key=True),
            Column('bid_prices', sqlalchemy.String),
            Column('ask_prices', sqlalchemy.String),
        ]
        if 'storage' in self.__participant:
            columns.extend([Column('storage_prices', sqlalchemy.String),
                            Column('storage_qtys', sqlalchemy.String)])

        table = sqlalchemy.Table(
            table_name,
            MetaData(),
            *columns
        )
        return table

    async def load_weights(self, **kwargs):
        self.status['weights_loading'] = True
        table_name = '_'.join((str(kwargs['generation']), kwargs['market_id'], 'weights', self.__participant['id']))
        db = dataset.connect(kwargs['db_path'])
        weights_table = db[table_name]
        weights = weights_table.find_one(generation=kwargs['generation'])
        if weights is not None:
            self.bid_prices = ast.literal_eval(weights['bid_prices'])
            self.ask_prices = ast.literal_eval(weights['ask_prices'])
            if 'storage' in self.__participant:
                self.storage_prices = ast.literal_eval(weights['storage_prices'])
                self.storage_qtys = ast.literal_eval(weights['storage_qtys'])

            self.status['weights_loading'] = False
            return True

        self.status['weights_loading'] = False
        return False

    async def step(self):
        next_actions = await self.act()
        await self.learn()
        if self.track_metrics:
            await self.metrics.save(10000)
        return next_actions

    # async def reset(self, **kwargs):
    #     return True