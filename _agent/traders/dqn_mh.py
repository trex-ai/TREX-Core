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

from itertools import product
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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

        # Generate actions
        self.actions = {
            'price': sorted(list(np.round(np.linspace(bid_price, ask_price, 9), 4))),
            'quantity': [int(q) for q in list(set(np.floor(np.linspace(-17, 17, 7))))]
            # 'quantity': [-17, ]
            # 'quantity': [-17, 17]
        }
        if 'storage' in self.__participant:
            # self.storage_type = self.__participant['storage']['type'].lower()
            self.actions['storage'] = self.actions['quantity']

        self.model = self.__create_model()
        self.model_target = self.__create_model()
        self.model_target.set_weights(self.model.get_weights())

        # Initialize learning parameters
        self.learning = kwargs['learning'] if 'learning' in kwargs else False
        reward_function = kwargs['reward_function'] if 'reward_function' in kwargs else None
        if reward_function:
            self._rewards = importlib.import_module('_agent.rewards.' + reward_function).Reward(
                self.__participant['timing'],
                self.__participant['ledger'],
                self.__participant['market_info'])

        self.learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs else 1e-6
        self.discount_factor = kwargs['discount_factor'] if 'discount_factor' in kwargs else 0.99
        self.exploration_factor = kwargs['exploration_factor'] if 'exploration_factor' in kwargs else 0.1

        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_function = keras.losses.Huber()
        self.action_history = dict()
        self.state_history = dict()
        self.rewards_history = dict()
        self.episode_reward = 0
        # self.episode_reward_history = []
        self.steps = 0

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

    def __create_model(self):
        num_inputs = 4 if 'storage' not in self.actions else 6
        num_hidden = 150 if 'storage' not in self.actions else 300
        # num_hidden = 64
        # num_hidden = 150
        # num_hidden = 300
        inputs = layers.Input(shape=(num_inputs,))
        hidden_1 = layers.Dense(num_hidden, activation="relu")(inputs)
        hidden_2 = layers.Dense(num_hidden, activation="relu")(hidden_1)
        hidden_3 = layers.Dense(num_hidden, activation="relu")(hidden_2)

        if 'storage' in self.__participant:
            hidden_4 = layers.Dense(num_hidden, activation="relu")(hidden_3)
            price = layers.Dense(len(self.actions['price']), activation="softmax")(hidden_4)
            quantity = layers.Dense(len(self.actions['quantity']), activation="softmax")(hidden_4)
            storage = layers.Dense(len(self.actions['storage']), activation="softmax")(hidden_4)
            return keras.Model(inputs=inputs, outputs=[price, quantity, storage])

        price = layers.Dense(len(self.actions['price']), activation="softmax")(hidden_3)
        quantity = layers.Dense(len(self.actions['quantity']), activation="softmax")(hidden_3)
        return keras.Model(inputs=inputs, outputs=[price, quantity])
        # return keras.Model(inputs=inputs, outputs={'price': price, 'quantity': quantity})

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

    # Core Functions, learn and act, called from outside
    async def learn(self, **kwargs):
        if not self.learning:
            return

        reward = await self._rewards.calculate()
        if reward is None:
            await self.metrics.track('rewards', reward)
            # del self.state_history[-1]
            # del self.action_history[-1]
            return

        current_round = self.__participant['timing']['current_round']
        next_settle = self.__participant['timing']['next_settle']
        round_duration = self.__participant['timing']['duration']
        # align reward with action timing
        # in the current market setup the reward is for actions taken 3 steps ago
        # if self._rewards.type == 'net_profit':
        reward_time_offset = current_round[1] - next_settle[1] - round_duration
        # print(reward_time_offset)
        self.rewards_history[current_round[1] + reward_time_offset] = reward
        # self.rewards_history.append((current_round[1] - 180, reward))
        self.episode_reward += reward
        await self.metrics.track('rewards', reward)

        # elif self._rewards.type == 'market_profit':
        #     timestamp = current_round[1]-180
        #     step_r = 0
        #     for r in reward:
        #         price_idx = self.actions['price'].index(r[0])
        #         quantity_idx = self.actions['quantity'].index(r[1])
        #         self.action_history[timestamp] = {
        #             'price': price_idx,
        #             'quantity': quantity_idx
        #         }
        #         self.rewards_history[timestamp] = r[2]
        #         step_r += r[2]
        #     self.episode_reward += step_r
        #     await self.metrics.track('rewards', step_r)

        if self.steps and len(self.rewards_history) >= 60 and not self.steps % 5:
            # TODO: change sampling to use dictionaries so timestamps line up properly
            common_ts = list(set(list(self.rewards_history.keys())).intersection(list(self.state_history.keys())[:-2]))
            indices = utils.secure_random.sample(common_ts, len(common_ts))

            state_sample = np.array([self.state_history[k] for k in indices])
            state_next_sample = np.array([self.state_history[k + round_duration] for k in indices])
            rewards_sample = [self.rewards_history[k] for k in indices]
            action_sample = [self.action_history[k] for k in indices]

            # print('----------')
            # print(state_sample)
            # print(state_next_sample)
            # print(action_sample)
            # print(rewards_sample)

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            # future_rewards = self.model_target.predict(state_next_sample, batch_size=int(0.05 * len(rewards_sample)))
            future_rewards = self.model_target.predict(state_next_sample, batch_size=min(30, int(0.05 * len(rewards_sample))))
            losses = [None] * len(self.actions)
            #TODO: repeat q update -> apply gradient for every head
            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = self.model(state_sample)
                for action in self.actions:
                    action_key_idx = list(self.actions.keys()).index(action)
                    updated_q_values = rewards_sample + self.discount_factor * tf.reduce_max(future_rewards[action_key_idx], axis=1)
                    num_actions = len(self.actions[action])

                    # Create a mask so we only calculate loss on the updated Q-values
                    action_sample_a = [i[action] for i in action_sample]
                    masks = tf.one_hot(action_sample_a, num_actions)

                    # Apply the masks to the Q-values to get the Q-value for action taken
                    q_action = tf.reduce_sum(tf.multiply(q_values[action_key_idx], masks), axis=1)
                    # Calculate loss between new Q-value and old Q-value
                    loss = self.loss_function(updated_q_values, q_action)
                    losses[action_key_idx] = loss

            # Backpropagation
            grads = tape.gradient(losses, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            # if self.steps >= 120:
            if self.steps >= 60 * 3:
                self.model_target.set_weights(self.model.get_weights())
                self.steps = 0
                self.rewards_history.clear()
                self.state_history.clear()
                self.action_history.clear()

            # if len(self.rewards_history) > 5000:
            #     self.rewards_history = self.rewards_history[1000:]
            #     self.state_history = self.state_history[1000:]
            #     self.action_history = self.action_history[1000:]

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
        current_round_end = utils.timestamp_to_local(current_round[1], timezone)
        # next_settle_end = utils.timestamp_to_local(next_settle[1], timezone)

        # state = [current_round_end.hour,
        #          current_round_end.minute,
        #          next_generation,
        #          next_load]

        state = [np.sin(2 * np.pi * current_round_end.hour / 24),
                 np.sin(2 * np.pi * current_round_end.minute / 60),
                 # np.cos(2 * np.pi * current_round_end.hour / 24),
                 # np.cos(2 * np.pi * current_round_end.minute / 60),
                 next_generation,
                 next_load]

        if 'storage' in self.__participant:
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
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = self.model(state_tensor, training=False)

            price_key_idx = list(self.actions.keys()).index('price')
            quantity_key_idx = list(self.actions.keys()).index('quantity')
            price_idx = tf.argmax(action_probs[price_key_idx][0]).numpy()
            quantity_idx = tf.argmax(action_probs[quantity_key_idx][0]).numpy()

            action_indices['price'] = price_idx
            action_indices['quantity'] = quantity_idx

            if 'storage' in self.actions:
                storage_key_idx = list(self.actions.keys()).index('storage')
                storage_idx = tf.argmax(action_probs[storage_key_idx][0]).numpy()
                action_indices['storage'] = storage_idx

                # TODO; fun experiments
                # if self.actions['storage'][storage_idx] < 0:
                #     action_indices['quantity'] = storage_idx

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
        self.steps += 1
        return next_actions

    async def end_of_generation_tasks(self):
        # self.episode_reward_history.append(self.episode_reward)
        self.model_target.set_weights(self.model.get_weights())
        print(self.__participant['id'], 'episode reward:', self.episode_reward)

    #     # update the the target network with new weights
    #     # self.model_target.set_weights(self.model.get_weights())
    #     # Limit the state and reward history
    #     if len(self.rewards_history) > 1440:
    #         del self.rewards_history[:1]
    #         del self.state_history[:1]
    #         del self.action_history[:1]

    async def reset(self, **kwargs):
        self.episode_reward = 0
        self.steps = 0
        self.rewards_history.clear()
        self.state_history.clear()
        self.action_history.clear()
        return True