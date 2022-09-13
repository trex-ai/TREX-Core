
import asyncio
import importlib
import os
import random
from collections import OrderedDict

import numpy as np
from _agent._utils.metrics import Metrics
from _utils import utils
from _utils.drl_utils import robust_argmax
from _utils.drl_utils import PPO_ExperienceReplay, EarlyStopper, huber, tb_plotter
import asyncio
from matplotlib import pyplot as plt
import sqlalchemy
from sqlalchemy import MetaData, Column
import dataset
import ast

from itertools import product
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras as k

class Trader:
    """This trader is not interacting with the local market (no bids/asks) and instead just attempts to follow a greedy battery strategy,
    maximizing self-consumption using a battery where possible"""

    def __init__(self, bid_price, ask_price, **kwargs):
        # Some utility parameters
        self.__participant = kwargs['trader_fns']
        self.publicparticipant = self.__participant
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
        self.actions = {}
        if 'P_max' in kwargs:
            p_max = kwargs['P_max']
        else:
            p_max = 17
        for action in kwargs['actions']:
            if action == 'price':
                self.actions['price'] = {'min': ask_price, 'max': bid_price}
            if action == 'quantity':
                self.actions['quantity'] = {'min': -p_max, 'max': p_max}
            if action == 'storage':
                self.actions['storage'] = {'min': -p_max, 'max': p_max}


        # initialize all the counters we need
        self.train_step = 0
        self.total_step = 0
        self.gen = 0

        #prepare TB functionality, to open TB use the terminal command: tensorboard --logdir <dir_path>
        cwd = os.getcwd()
        experiment_path = os.path.join(cwd, kwargs['study_name'])
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

        #ToDo: test if having parameter sharing helps here?
        #logs we need for plotting
        self.rewards_history = []
        self.state_history = []
        self.net_load_history = []
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

        if mode not in ('subtract', 'multiply', 'set'):
            return False

        param_value = getattr(self, parameter)
        if mode == 'subtract':
            param_value = max(0, param_value - adjustment)

        elif mode == 'multiply':
            param_value *= adjustment

        elif mode == 'set':
            param_value = adjustment

        if limit is not None:
            param_value = max(param_value, limit)

        setattr(self, parameter, param_value)

    # Core Functions, learn and act, called from outside
    async def learn(self, **kwargs):
        # print(self.total_step)
        if not self.learning:
            return
        current_round = self.__participant['timing']['current_round']
        next_settle = self.__participant['timing']['next_settle']
        round_duration = self.__participant['timing']['duration']

        reward = await self._rewards.calculate()
        if reward is None:
            await self.metrics.track('rewards', reward)
            return
        # align reward with action timing
        # in the current market setup the reward is for actions taken 3 steps ago
        # if self._rewards.type == 'net_profit':
        reward_time_offset = current_round[1] - next_settle[1] - round_duration
        reward_timestamp = current_round[1] + reward_time_offset

        await self.metrics.track('rewards', reward)
        self.rewards_history.append(reward)

    async def act(self, **kwargs):
        # Generate state (inputs to model):
        # - time(s)
        # - next generation
        # - next load
        # - battery stats (if available)

        current_round = self.__participant['timing']['current_round']
        next_settle = self.__participant['timing']['next_settle']
        next_generation, next_load = await self.__participant['read_profile'](next_settle)
        timezone = self.__participant['timing']['timezone']

        minutes = int(current_round[0]/60)
        sin_24 = np.sin(2*np.pi*minutes/24) #ToDo: ATM THIS IS ONLY FOR THE FAKE 24H synth profile!!
        cos_24 = np.cos(2 * np.pi * minutes / 24)  # ToDo: ATM THIS IS ONLY FOR THE FAKE 24H synth profile!!
        pseudohour = minutes%24
        state = [
                 # np.sin(2 * np.pi * current_round_end.hour / 24),
                 # np.cos(2 * np.pi * current_round_end.hour / 24),
                 # np.sin(2 * np.pi * current_round_end.minute / 60),
                 # np.cos(2 * np.pi * current_round_end.minute / 60),
                 sin_24, cos_24,
                 float(next_generation/17),
                 float(next_load/17)]
        # print('gen', next_generation, 'load', next_load, 'time', current_round[0])

        if 'storage' in self.__participant:
            storage_schedule = await self.__participant['storage']['check_schedule'](current_round)
            soc = storage_schedule[current_round]['projected_soc_end']
            battery_out_current = storage_schedule[current_round]['energy_scheduled']
            state.append(soc)
        state = np.array(state)
        self.state_history.append(state)

        #ToDO: check this for validity
        # It makes sense that it should be the current SoC that we are observing, not the future SoC
        #goal: 0 = next_load - next_gen - battery_target
        battery_target = -(next_load - next_generation)
        target_action = {}
        target_action['storage'] = battery_target
        actions = await self.decode_actions(target_action, next_settle)

        current_generation, current_load = await self.__participant['read_profile'](current_round)
        if 'storage' in self.__participant:
            net_load_current = current_load - current_generation + storage_schedule[current_round]['energy_scheduled']
        else:
            net_load_current = current_load - current_generation
        self.net_load_history.append(net_load_current)

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


        if 'storage' in taken_action:
            storage = int(taken_action['storage'])


        if 'storage' in self.actions:
            actions['bess'] = {
                str(next_settle): storage
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
        # print(self.__participant['id'], 'episode reward:', episode_G)

        data_for_tb = [{'name':'Return', 'data':episode_G, 'type':'scalar', 'step':self.gen},
                       {'name': 'Episode Rewards', 'data': self.rewards_history, 'type': 'histogram', 'step':self.gen},
                       ]
        for action in self.actions:
            data_for_tb.append({'name':action, 'data':self.actions_history[action], 'type':'histogram', 'step':self.gen})

        state_history = np.array(self.state_history)
        state_history = state_history[8:,:]
        # plt.plot(state_history[0:24,0])
        # plt.show()
        # plt.plot(state_history[0:24, 1])
        # plt.show()
        # plt.plot(state_history[0:24, 2])
        # plt.show()
        # plt.plot(state_history[0:24, 3])
        # plt.show()
        # plt.plot(state_history[0:24, 4])
        # plt.show()

        day_length = 24
        socs = state_history[:,-1]*100
        full_days = socs.shape[-1]%24

        data_for_tb.append({'name': 'SoC_during_day', 'data': socs, 'type': 'pseudo3D', 'step':self.gen, 'buckets': day_length})

        net_load_history = self.net_load_history - np.amin(self.net_load_history)
        net_load_history = net_load_history[8:]
        data_for_tb.append(
            {'name': 'Effective_Ned_load_during_day', 'data': net_load_history, 'type': 'pseudo3D', 'step': self.gen, 'buckets': day_length})

        # loop = asyncio.get_running_loop()
        # await loop.run_in_executor(None, tb_plotter, data_for_tb, self.summary_writer)
        tb_plotter(data_for_tb, self.summary_writer)


        self.gen = self.gen + 1

    async def reset(self, **kwargs):

        self.rewards_history.clear()
        self.state_history.clear()
        self.net_load_history.clear()
        for action in self.actions:
            self.actions_history[action].clear()

        return True