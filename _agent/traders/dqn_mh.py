"""This implements a tabular epsilon greedy Q-learning for a MicroTE Bandit problem

"""
import asyncio
import importlib
import os
import random
from collections import OrderedDict

import numpy as np
from TREX_Core._agent._utils.metrics import Metrics
from TREX_Core._utils import utils
from TREX_Core._utils.drl_utils import robust_argmax, ExperienceReplayBuffer

import sqlalchemy
from sqlalchemy import MetaData, Column
import dataset
import ast

from itertools import product
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorboard as tb


class Trader:
    """This trader uses SMA crossover to make trading decisions in the context of MicroFE

    The trader tries to learn the right prices for each minute of the day. This is done by initializing two prices tables, one for bid prices and one for ask prices. Each table is 1440 elements long. The tables are initialized by randomizing prices of each minute within a price range. A 15 minute window is used for initialization, which means that only 96 initial prices are generated. This is meant to decrease initial noise. Successful trades will nudge bid and ask prices to the point of most profit and least cost.
    """
    def __init__(self, bid_price, ask_price, **kwargs):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        # Some utility parameters
        self.__participant = kwargs['trader_fns']
        self.study_name = kwargs['study_name'] if 'study_name' in kwargs else None
        self.status = {
            'weights_loading': False
        }

        # Initialize metrics tracking
        self.track_metrics = kwargs['track_metrics'] if 'track_metrics' in kwargs else False
        self.metrics = Metrics(self.__participant['id'], track=self.track_metrics)
        if self.track_metrics:
            self.__init_metrics()

        # Generate actions
        self.actions = {
            'price': sorted(list(np.round(np.linspace(bid_price, ask_price, 7), 4))),
            'quantity': [int(q) for q in list(set(np.round(np.linspace(-17, 17, 7))))]
            # 'quantity': [-17, ]
            # 'quantity': [-17, 17]
        }
        if 'storage' in self.__participant:
            # self.storage_type = self.__participant['storage']['type'].lower()
            self.actions['storage'] = self.actions['quantity']

        # initialize all the counters we need
        self.train_step = 0
        self.total_step = 0
        self.gen = 0

        #prepare TB functionality, to open TB use the terminal command: tensorboard --logdir <dir_path>
        cwd = os.getcwd()
        logs_path = os.path.join(cwd, 'Logs')
        experiment_path = os.path.join(logs_path, self.study_name)
        trader_path = os.path.join(experiment_path, self.__participant['id'])

        self.summary_writer = tf.summary.create_file_writer(trader_path)
        self.episode_actions = {}
        for action in self.actions:
            self.episode_actions[action] = []

        self.model = self.__create_model()
        self.model_target = self.__create_model()
        # self.model_target.set_weights(self.model.get_weights())

        # Initialize learning parameters
        self.learning = kwargs['learning'] if 'learning' in kwargs else False
        reward_function = kwargs['reward_function'] if 'reward_function' in kwargs else None
        if reward_function:
            self._rewards = importlib.import_module('TREX_Core._agent.rewards.' + reward_function).Reward(
                self.__participant['timing'],
                self.__participant['ledger'],
                self.__participant['market_info'])

        #DQN hyperparameters:
        self.learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs else 1e-4 # Original DQN uses RMSProp, learning rate of α = 0.00025, Rainbow’s variants used a learning rate of α/4, selected among {α/2, α/4, α/6}, and a value of 1.5 × 10−4 for Adam’s  hyper-parameter,
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, epsilon=1.5*10e-4) # Adam over RMSprop because apparently less sensitive to choice of α (see Rainbow)
        self.loss_function = keras.losses.Huber()  # Classic Huber loss, this is known to be a straight improvement over square loss
        self.batch_size = 128 #bigger is smoother, but might require a bigger replay buffer
        self.discount_factor = kwargs['discount_factor'] if 'discount_factor' in kwargs else 0.99
        self.exploration_factor = kwargs['exploration_factor'] if 'exploration_factor' in kwargs else 0.5 #DQN and Rainbow both start with epsilon 1 and decrease to 0.1-0.01 relatively fast with subsequent annealing down.
        # Replay buffer initialization, Rainbow mentions default value of 200K replay buffer filling BEFORE learning!
        # ToDO: Rewrite replay bufffer completely into a separate class/object
        self.backprop_frequency = 4 #figure out if we really need this?
        buffer_length_base = 20*1440
        self.tau = 1
        self.target_network_update_frequency = int(buffer_length_base/3)
        self.experience_replay_buffer = ExperienceReplayBuffer(max_length=buffer_length_base, #Setting this to a multiple of the batch size and then mulitply with the number of days in one gen for now
                                                               learn_wait=int(buffer_length_base/3))

        self.ddqn = True

        self.action_history = OrderedDict()
        self.state_history = OrderedDict()
        self.rewards_history = []



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

    def __create_model(self):
        num_inputs = 2 if 'storage' not in self.actions else 8
        num_hidden = 64 if 'storage' not in self.actions else 300
        num_hidden_layers = 2 #lets see how far we get with this first
        # num_hidden = 64
        # num_hidden = 128
        # num_hidden = 256
        initializer = tf.keras.initializers.HeNormal()
        Q = {}
        inputs = layers.Input(shape=(num_inputs,))
        internal_signal = inputs
        for hidden_layer_number in range(num_hidden_layers): #hidden layers
            internal_signal = layers.Dense(num_hidden,
                                           activation="elu",
                                           # bias_initializer=initializer,
                                           kernel_initializer=initializer)(internal_signal)

        value = layers.Dense(1)(internal_signal)
        # print('internal signal', internal_signal)
        for action in self.actions:

            print(' in the for loop', action, len(self.actions[action]))
            advantage = layers.Dense(len(self.actions[action]))(internal_signal)
            print(tf.reduce_mean(advantage, axis=-1, keepdims=True))
            Q[action] = value + advantage - tf.reduce_mean(advantage, axis=-1, keepdims=True)
        # print('This is the Q dictionary', Q)
        return keras.Model(inputs=inputs, outputs=Q)

    def __create_model_GRU(self):
        num_inputs = 2 if 'storage' not in self.actions else 8
        num_hidden = 64 if 'storage' not in self.actions else 300
        num_hidden_layers = 2 #lets see how far we get with this first
        # num_hidden = 64
        # num_hidden = 128
        # num_hidden = 256
        initializer = tf.keras.initializers.HeNormal()
        Q = {}
        inputs = layers.Input(shape=(num_inputs,)) #shape inputs is [batch,observations]
        inputs = tf.expand_dims(inputs, axis=1) #shape inputs is [batch,1,observations]

        internal_signal = inputs
        # for hidden_layer_number in range(num_hidden_layers): #hidden layers
        #     internal_signal = layers.Dense(num_hidden,
        #                                    activation="elu",
        #                                    # bias_initializer=initializer,
        #                                    kernel_initializer=initializer)(internal_signal)

        internal_signal = layers.GRU(num_hidden, return_sequences=True)(internal_signal)
        value = layers.Dense(1)(internal_signal)
        for action in self.actions:

            advantage = layers.Dense(len(self.actions[action]))(internal_signal)
            Q[action] = value + advantage - tf.reduce_mean(advantage, axis=-1, keepdims=True) #shape [batch, 1, actions]
            Q[action] = tf.squeeze(Q[action], axis=1 )# shape [batch, actions]

        return keras.Model(inputs=inputs, outputs=Q)

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

        param_value = max(param_value, 0.05)

        setattr(self, parameter, param_value)

    # Core Functions, learn and act, called from outside
    async def learn(self, **kwargs):
        if not self.learning:
            return
        current_round = self.__participant['timing']['current_round']
        next_settle = self.__participant['timing']['next_settle']
        round_duration = self.__participant['timing']['duration']

        reward = await self._rewards.calculate()
        if reward is None:
            await self.metrics.track('rewards', reward)
            # del self.state_history[-1]
            # del self.action_history[-1]
            return
        # align reward with action timing
        # in the current market setup the reward is for actions taken 3 steps ago
        # if self._rewards.type == 'net_profit':
        reward_time_offset = current_round[1] - next_settle[1] - round_duration
        reward_timestamp = current_round[1] + reward_time_offset

        await self.metrics.track('rewards', reward)
        self.rewards_history.append(reward)

        with self.summary_writer.as_default():
            tf.summary.scalar('reward', reward, step=self.total_step)

        if reward_timestamp in self.state_history and reward_timestamp in self.action_history:  # we found matching ones, buffer and pop
            self.experience_replay_buffer.add_entry(ts=reward_timestamp,
                                                    rewards=reward,
                                                    actions=self.action_history[reward_timestamp],
                                                    states=self.state_history[reward_timestamp],
                                                    episode=self.gen)

            self.action_history.pop(reward_timestamp)
            self.state_history.pop(reward_timestamp)

        if self.experience_replay_buffer.should_we_learn() and not self.total_step % self.backprop_frequency:
            batch = self.experience_replay_buffer.fetch_batch(batchsize=self.batch_size)

            # states_equal = np.all(np.equal(state_sample, batch['states']))
            # next_states_equal = np.all(np.equal(state_next_sample, batch['next_states']))
            # rewards_equal = np.all(np.equal(rewards_sample, batch['rewards']))

            q_bootstrap = self.model_target(batch['next_states'], training=False)
            if self.ddqn:
                q_bootstrap_choice = self.model(batch['next_states'], training=False)

            losses = []
            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = self.model(batch['states'])
                for action in self.actions:
                    #Get the bootstrapping target
                    if self.ddqn:
                        # index = robust_argmax(q_bootstrap_choice[action])
                        indices = []
                        for sample in range(self.batch_size):
                            values_samples = tf.expand_dims(q_bootstrap_choice[action][sample,:], axis=0)
                            max_index = await robust_argmax(values_samples)
                            indices.append([sample,  max_index.numpy()[0]])

                        q_target = tf.gather_nd(q_bootstrap[action], indices)
                        # index = index.numpy().tolist()
                        # max_values = tf.reduce_max(q_bootstrap_choice[action], axis=-1)
                        # max_values = max_values.numpy().tolist()
                        # index = 0
                        # max_value_indices = []
                        # for sample in range(self.batch_size):
                        #     sample_max_index = tf.where(tf.math.equal(max_values[sample], q_bootstrap_choice[action][sample,:]))
                        #     sample_max_index = tf.random.shuffle(sample_max_index)[0]
                        #
                        #     max_value_indices.append([sample, sample_max_index.numpy().tolist()])

                    else:
                        q_target = tf.reduce_max(q_bootstrap[action], axis=-1)

                    q_target = batch['rewards'] + self.discount_factor * q_target

                    # Create a mask so we only calculate loss on the chosen action's  Q-values
                    # Apply the masks to the Q-values to get the Q-value for action taken
                    q_action = q_values[action]
                    # action_sample_action = [i[action] for i in action_sample]
                    action_batch = [i[action] for i in batch['actions']]
                    # actions_equal = np.all(np.equal(action_sample_action, action_batch))
                    masks = tf.one_hot(action_batch,
                                       len(self.actions[action]))
                    q_action_chosen = tf.reduce_sum(q_action * masks, axis=1)

                    # Calculate loss between new Q-value and old Q-value, append
                    loss = self.loss_function(q_target, q_action_chosen)
                    losses.append(loss)

            # Backpropagation
            grads = tape.gradient(losses, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            #clearing replay buffer
            # if len(self.rewards_history) > self.replay_buffer_length:
            #     diff = len(self.rewards_history) - self.replay_buffer_length
            #     for times in range(diff):
            #          self.rewards_history.popitem(last=False)
            #          self.state_history.popitem(last=False)
            #          self.action_history.popitem(last=False)

            # logging graphs
            with self.summary_writer.as_default():
                self.train_step = self.train_step + 1
                for action in self.actions.keys():
                    action_key_idx = list(self.actions.keys()).index(action)
                    tf.summary.scalar('loss_'+action, losses[action_key_idx], step=self.train_step)

        #updating target network, if tau smaller one perform soft update
        if not self.total_step % self.target_network_update_frequency:
            if self.tau < 1:
                target_network_w = [target_weight * (1 - self.tau) + q_weight * self.tau for q_weight, target_weight in zip(self.model.get_weights(), self.model_target.get_weights())]
                self.model_target.set_weights(target_network_w)
            else:
                self.model_target.set_weights(self.model.get_weights())

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
        # current_round_end = utils.timestamp_to_local(current_round[1], timezone)
        # next_settle_end = utils.timestamp_to_local(next_settle[1], timezone)

        state = [
                 # np.sin(2 * np.pi * current_round_end.hour / 24),
                 # np.cos(2 * np.pi * current_round_end.hour / 24),
                 # np.sin(2 * np.pi * current_round_end.minute / 60),
                 # np.cos(2 * np.pi * current_round_end.minute / 60),
                 next_generation,
                 next_load]

        if 'storage' in self.__participant: #ToDo: Access SoC
            storage_schedule = await self.__participant['storage']['check_schedule'](next_settle)
            # storage_schedule = self.__participant['storage']['schedule'](next_settle)
            max_charge = storage_schedule[next_settle]['energy_potential'][1]
            max_discharge = storage_schedule[next_settle]['energy_potential'][0]
            state.extend([max_charge, max_discharge])

        state = np.array(state)
        epsilon = self.exploration_factor if self.learning else -1
        explore = utils.secure_random.random() <= epsilon

        action_indices = dict()
        if explore:
            for action in self.actions:
                action_indices[action] = utils.secure_random.choice(range(len(self.actions[action])))
        else:
            # action_probs, critic_value = self.model(state, training=False)
            state_tensor = tf.expand_dims(tf.convert_to_tensor(state), 0)
            action_values = self.model(state_tensor, training=False)

            action_indices = {}
            for action in self.actions:
                action_idx = await robust_argmax(action_values[action])
                action_indices[action] = action_idx.numpy()[0]


            with self.summary_writer.as_default():
                # this reduces across all dimensions, we assume only one action selection per TS
                for action in self.actions:
                    tf.summary.histogram('Q_'+str(action),
                                          action_values[action],
                                          step=self.total_step)
                    # tf.summary.scalar(str(action),
                    #                   self.actions[action][action_indices[action]],
                    #                   step=self.total_step)


        # with self.gradient_tape:
        actions = await self.decode_actions(action_indices, next_settle)
        # print(state)
        self.state_history[current_round[1]] = state
        self.action_history[current_round[1]] = action_indices
        # self.state_history.append((current_round[1], state))
        # self.action_history.append((current_round[1], action_indices))
        # self.critic_value_history.append(critic_value[0, 0])

        if self.track_metrics:
            await asyncio.gather(
                self.metrics.track('timestamp', self.__participant['timing']['current_round'][1]),
                self.metrics.track('actions_dict', actions),
                self.metrics.track('next_settle_load', next_load),
                self.metrics.track('next_settle_generation', next_generation))
            if 'storage' in self.actions:
                await self.metrics.track('storage_soc', self.__participant['storage']['info']()['state_of_charge'])
        return actions

    async def decode_actions(self, action_indices: dict, next_settle):
        actions = dict()
        # print(action_indices)

        price = self.actions['price'][action_indices['price']]
        quantity = self.actions['quantity'][action_indices['quantity']]

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
            target = self.actions['storage'][action_indices['storage']]
            if target:
                actions['bess'] = {
                    str(next_settle): target
                }
        # print(actions)

        #log actions for later histogram plot
        for action in self.actions:
            self.episode_actions[action].append(self.actions[action][action_indices[action]])
        return actions
    # async def save_model(self, **kwargs):
    #     '''
    #     Save the price tables at the end of the episode into database
    #     '''
    #
    #     table_name = '_'.join((str(kwargs['generation']), kwargs['market_id'], 'weights', self.__participant['id']))
    #     table = self.__create_weights_table(table_name)
    #     await db_utils.create_table(db_string=kwargs['db_path'],
    #                                 table_type='custom',
    #                                 custom_table=table)
    #     weights = [{
    #         'generation': kwargs['generation'],
    #         'bid_prices': str(self.bid_prices),
    #         'ask_prices': str(self.ask_prices)
    #     }]
    #     await db_utils.dump_data(weights, kwargs['db_path'], table)
    #
    # def __create_weights_table(self, table_name):
    #     columns = [
    #         Column('generation', sqlalchemy.Integer, primary_key=True),
    #         Column('bid_prices', sqlalchemy.String),
    #         Column('ask_prices', sqlalchemy.String),
    #     ]
    #     table = sqlalchemy.Table(
    #         table_name,
    #         MetaData(),
    #         *columns
    #     )
    #     return table
    #
    # async def load_model(self, **kwargs):
    #     self.status['weights_loading'] = True
    #     table_name = '_'.join((str(kwargs['generation']), kwargs['market_id'], 'weights', self.__participant['id']))
    #     db = dataset.connect(kwargs['db_path'])
    #     weights_table = db[table_name]
    #     weights = weights_table.find_one(generation=kwargs['generation'])
    #     if weights is not None:
    #         self.bid_prices = ast.literal_eval(weights['bid_prices'])
    #         self.ask_prices = ast.literal_eval(weights['ask_prices'])
    #         self.status['weights_loading'] = False
    #         return True
    #
    #     self.status['weights_loading'] = False
    #     return False

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
        self.model_target.set_weights(self.model.get_weights())
        print(self.__participant['id'], 'episode reward:', episode_G)

        if self.gen > 30:
            self.learning_rate = 0.9 * self.learning_rate
            self.learning_rate = max(self.learning_rate, 1e-7)
            self.optimizer.lr.assign(self.learning_rate)

        with self.summary_writer.as_default():
            tf.summary.scalar('Return' , episode_G, step= self.gen)
            tf.summary.histogram('Rewards during Episode', self.rewards_history, step=self.gen)
            for action in self.actions:
                tf.summary.histogram(action, self.episode_actions[action], step=self.gen)
        self.gen = self.gen + 1

    async def reset(self, **kwargs):
        self.state_history.clear()
        self.action_history.clear()
        self.rewards_history = []
        for action in self.actions:
            self.episode_actions[action] = []

        return True