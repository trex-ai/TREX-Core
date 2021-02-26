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

        # Initialize the agent learning parameters
        self.bid_prices = self.__generate_price_table(bid_price, ask_price, 100)
        self.ask_prices = self.__generate_price_table(bid_price, ask_price, 100)
        self.bid_price = utils.secure_random.choice(list(self.bid_prices.keys()))
        self.ask_price = utils.secure_random.choice(list(self.ask_prices.keys()))
        # Initialize learning parameters
        self.learning = kwargs['learning'] if 'learning' in kwargs else False
        reward_function = kwargs['reward_function'] if 'reward_function' in kwargs else None
        if reward_function:
            self._rewards = importlib.import_module('_agent.rewards.' + reward_function).Reward(
                self.__participant['timing'],
                self.__participant['ledger'],
                self.__participant['market_info'])

        self.learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs else 0.1
        self.discount_factor = kwargs['discount_factor'] if 'discount_factor' in kwargs else 0.98
        self.epsilon = kwargs['epsilon'] if 'epsilon' in kwargs else 0.1

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

    # Core Functions, learn and act, called from outside
    async def learn(self, **kwargs):
        if not self.learning:
            return

        reward = await self._rewards.calculate()

        if reward is None:
            await self.metrics.track('rewards', reward)
            return

        await self.metrics.track('rewards', reward)

        q_bid = self.bid_prices[self.bid_price]
        q_max_bid = max(self.bid_prices.values())
        q_bid_new = q_bid + self.learning_rate * (reward + self.discount_factor * q_max_bid - q_bid)
        self.bid_prices[self.bid_price] = q_bid_new

        q_ask = self.ask_prices[self.ask_price]
        q_max_ask = max(self.ask_prices.values())
        q_ask_new = q_ask + self.learning_rate * (reward + self.discount_factor * q_max_ask - q_ask)
        self.ask_prices[self.ask_price] = q_ask_new

    async def act(self, **kwargs):
        actions = {}
        next_settle = self.__participant['timing']['next_settle']

        next_generation, next_load = await self.__participant['read_profile'](next_settle)
        next_residual_load = next_load - next_generation
        next_residual_gen = -next_residual_load

        epsilon = self.epsilon if self.learning else -1
        if utils.secure_random.random() <= epsilon:
            self.bid_price = utils.secure_random.choice(list(self.bid_prices.keys()))
            self.ask_price = utils.secure_random.choice(list(self.ask_prices.keys()))
        else:
            q_max_bid = max(self.bid_prices.values())
            q_max_ask = max(self.ask_prices.values())

            bid_argmaxes = [price for price, q in self.bid_prices.items() if q == q_max_bid]
            ask_argmaxes = [price for price, q in self.ask_prices.items() if q == q_max_ask]

            self.bid_price = utils.secure_random.choice(bid_argmaxes)
            self.ask_price = utils.secure_random.choice(ask_argmaxes)

        if next_residual_load > 0:
            actions['bids'] = {
                str(next_settle): {
                    'quantity': next_residual_load,
                    'source': 'solar',
                    'price': self.bid_price
                }
            }

        if next_residual_gen > 0:
            actions['asks'] = {
                str(next_settle): {
                    'quantity': next_residual_gen,
                    'source': 'solar',
                    'price': self.ask_price
                }
            }

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
        await db_utils.dump_data(weights, kwargs['db_path'], table)

    def __create_weights_table(self, table_name):
        columns = [
            Column('generation', sqlalchemy.Integer, primary_key=True),
            Column('bid_prices', sqlalchemy.String),
            Column('ask_prices', sqlalchemy.String),
        ]
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