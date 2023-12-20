
import asyncio
import importlib
import os
import random
from collections import OrderedDict

import numpy as np
from _agent._utils.metrics import Metrics
from _utils import utils
from _utils.drl_utils import robust_argmax
from _utils.drl_utils import PPO_ExperienceReplay, EarlyStopper
import asyncio

import sqlalchemy
from sqlalchemy import MetaData, Column
import dataset
import ast

from itertools import product
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import tensorboard as tb


class Trader:
    """This trader uses the proximal policy optimization algorithm (PPO) as proposed in https://arxiv.org/abs/1707.06347.
    Any liberties and further modifications to the algorithm will be attempted to be documented here
    Impelmentation is inspired by the torch implementation from cleanRL and by
    https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/PPO/tf2/agent.py
    """
    def __init__(self, bid_price, ask_price, **kwargs):
        # Some utility parameters
        self.__participant = kwargs['trader_fns']
        self.study_name = kwargs['study_name']
        self.status = {
            'weights_loading': False
        }

        # Initialize metrics tracking
        self.track_metrics = kwargs['track_metrics']
        self.metrics = Metrics(self.__participant['id'], track=self.track_metrics)
        if self.track_metrics:
            self.__init_metrics()

        # Generate actions
        #I think we could stay with quantized actions, however I'd like to start testing on the non-quantized version ASAP so we do non quantized
        self.actions = {
            'price': {'min': ask_price,
                      'max': bid_price},
           'quantity': {'min': kwargs['min_quantity'] if 'min_quantity' in kwargs else -17,
                         'max': kwargs['max_quantity'] if 'max_quantity' in kwargs else 17}
        }

        if 'storage' in self.__participant:
            # self.storage_type = self.__participant['storage']['type'].lower()
            self.actions['storage'] = self.actions['quantity']

        # initialize all the counters we
        self.total_step = 0
        self.gen = 0

        #prepare TB functionality, to open TB use the terminal command: tensorboard --logdir <dir_path>
        cwd = os.getcwd()
        logs_path = os.path.join(cwd, 'Battery_Test')
        experiment_path = os.path.join(logs_path, self.study_name)
        trader_path = os.path.join(experiment_path, self.__participant['id'])

        self.summary_writer = tf.summary.create_file_writer(trader_path)

        # Initialize learning parameters
        self.learning = kwargs['learning']
        reward_function = kwargs['reward_function']
        if reward_function:
            self._rewards = importlib.import_module('_agent.rewards.' + reward_function).Reward(
                self.__participant['timing'],
                self.__participant['ledger'],
                self.__participant['market_info'])

        self.ppo_actor_dist = tfp.distributions.Dirichlet(concentration =[2.0, 2.0],
                                                       validate_args=False,
                                                       allow_nan_stats=True,
                                                       name='Uniform' + self.__participant['id'])
        # self.ppo_actor_dist = tfp.distributions.Uniform(high=1.0,
        #                                                 low=0.0,
        #                                                 validate_args=False,
        #                                                 allow_nan_stats=True,
        #                                                 name='Uniform' + self.__participant['id'])
        # Buffers we need for logging stuff before putting into the PPo Memory
        self.actions_buffer = {}

        #logs we need for plotting
        self.rewards_history = []
        self.actions_history = {}
        for action in self.actions:
            self.actions_history[action] = []

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

    def anneal(self, parameter:str, adjustment, mode:str='multiply', limit=None):
        if not hasattr(self, parameter):
            return False

    # Core Functions, learn and act, called from outside
    async def learn(self, **kwargs):

        reward = await self._rewards.calculate()

        if reward is None:
            await self.metrics.track('rewards', reward)
            return
        else:
            self.rewards_history.append(reward)
            await self.metrics.track('rewards', reward)

    async def __sample_pi(self):

#        min = 1e-10
#        a_dist = tf.clip_by_value(a_dist, clip_value_min=min, clip_value_max=0.9999999)
#         a = self.ppo_actor_dist.sample()
#         a = tf.clip_by_value(a, clip_value_min=1e-10, clip_value_max=0.9999999)
#         a = a.numpy().tolist()
        a = self.ppo_actor_dist.sample()
        a = tf.clip_by_value(a, clip_value_min=1e-10, clip_value_max=0.9999999)
        a = a.numpy().tolist()
        a_scaled = {}
        keys = list(self.actions.keys())
        for action_index in range(len(keys)):
            a_index = a[action_index]
            min = self.actions[keys[action_index]]['min']
            max = self.actions[keys[action_index]]['max']
            a_index = min + (a_index * (max - min))
            a_scaled[keys[action_index]] = a_index

        return a_scaled

    async def act(self, **kwargs):

        current_round = self.__participant['timing']['current_round']
        next_settle = self.__participant['timing']['next_settle']
        next_generation, next_load = await self.__participant['read_profile'](next_settle)
        self.net_load = next_load - next_generation

        state = [float(next_generation/17),
                 float(next_load/17)]

        if 'storage' in self.__participant: #ToDo: Check if this is the right way to acess SOC
            storage_schedule = await self.__participant['storage']['check_schedule'](next_settle)
            soc = storage_schedule[next_settle]['projected_soc_end']
            state.append(soc)

        taken_action = await self.__sample_pi()

        actions = await self.decode_actions(taken_action, next_settle)

        if self.track_metrics:
            await asyncio.gather(
                self.metrics.track('timestamp', self.__participant['timing']['current_round'][1]),
                self.metrics.track('actions_dict', actions),
                self.metrics.track('next_settle_load', next_load),
                self.metrics.track('next_settle_generation', next_generation))
            if 'storage' in self.actions:
                await self.metrics.track('storage_soc', self.__participant['storage']['info']()['state_of_charge'])
        return actions

    async def decode_actions(self, taken_action, next_settle):
        actions = dict()

        if 'price' in taken_action:
            price = taken_action['price']
            price = round(price, 4)
        else:
            price = 0.11 #ToDO: ok default?

        if 'quantity' in taken_action:
            quantity = taken_action['quantity']
            quantity = int(quantity)
        else:
            quantity = self.net_load

        if quantity > 0:
            actions['bids'] = {
                str(next_settle): {
                    'quantity': quantity,
                    'price': price
                }
            }
        elif quantity < 0:
            actions['asks'] = {
                'solar': {
                    str(next_settle): {
                        'quantity': -quantity,
                        'price': price
                    }
                }
            }

        if 'storage' in self.actions:
            actions['bess'] = {
                str(next_settle): taken_action['storage']
                }
        # print(actions)

        #log actions for later histogram plot
        for action in self.actions:
            self.actions_history[action].append(taken_action[action])
        return actions

    async def step(self):
        next_actions = await self.act()
        await self.learn()
        if self.track_metrics:
            await self.metrics.save(10000)
        # print(next_actions)
        self.total_step += 1
        return next_actions

    async def end_of_generation_tasks(self):
        # self.episode_reward_history.append(self.episode_reward)
        episode_G = sum(self.rewards_history)
        print(self.__participant['id'], 'episode reward:', episode_G)

        with self.summary_writer.as_default():
            tf.summary.scalar('Return' , episode_G, step= self.gen)
            tf.summary.histogram('Rewards during Episode', self.rewards_history, step=self.gen)

            for action in self.actions:
                tf.summary.histogram(action, self.actions_history[action], step=self.gen)

        self.gen = self.gen + 1

    async def reset(self, **kwargs):
        self.actions_buffer.clear()

        self.rewards_history.clear()
        for action in self.actions:
            self.actions_history[action].clear()

        return True