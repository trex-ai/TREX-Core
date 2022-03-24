
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
from tensorflow.keras import layers
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

        # initialize all the counters we need
        self.train_step = 0
        self.total_step = 0
        self.gen = 0

        #prepare TB functionality, to open TB use the terminal command: tensorboard --logdir <dir_path>
        cwd = os.getcwd()
        logs_path = os.path.join(cwd, 'Battery_Test')
        experiment_path = os.path.join(logs_path, self.study_name)
        trader_path = os.path.join(experiment_path, self.__participant['id'])

        self.summary_writer = tf.summary.create_file_writer(trader_path)

        # Initialize learning parameters
        self.learning = kwargs['learning'] if 'learning' in kwargs else False
        reward_function = kwargs['reward_function'] if 'reward_function' in kwargs else None
        if reward_function:
            self._rewards = importlib.import_module('_agent.rewards.' + reward_function).Reward(
                self.__participant['timing'],
                self.__participant['ledger'],
                self.__participant['market_info'])

        #ToDo: test if having parameter sharing helps here?

        # Hyperparameters
        self.alpha_critic = kwargs['alpha_critic'] if 'alpha_critic' in kwargs else 1e-3
        self.alpha_actor = kwargs['alpha_actor'] if 'alpha_actor' in kwargs else 3*1e-4
        self.batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 128 #bigger is smoother, but might require a bigger replay buffer
        self.policy_clip = kwargs['policy_clip'] if 'policy_clip' in kwargs else 0.2#ToDo: look into some type of entropy regularization to guide exploration!
        self.max_train_steps = kwargs['max_train_steps'] if 'max_train_steps' in kwargs else 100 #ToDO: figure out reasonable default valye and make this a hyperparam
        self.kl_stop = kwargs['kl_stop'] if 'kl_stop' in kwargs else 0.03 #according to baselines tends to be bewteen 0.01 and 0.05
        self.critic_patience = kwargs['critic_patience'] if 'critic_patience' in kwargs else int(self.max_train_steps/10)
        self.entropy_reg = kwargs['entropy_reg'] if 'entropy_reg' in kwargs else 0.1
        self.gamma = kwargs['gamma'] if 'gamma' in kwargs else 0.99
        self.gae_lambda = kwargs['gae_lambda'] if 'gae_lambda' in kwargs else 0.95
        self.normalize_advantages = kwargs['normalize_advantages'] if 'normalize_advantages' in kwargs else True
        self.ppo_actor = self.__build_actor()
        self.ppo_actor.compile(optimizer=keras.optimizers.Adam(learning_rate=self.alpha_actor,
                                                               # epsilon=1.5*10e-4 #ToDo: research usual ADAM hyperparams for PPO types
                                                               ),
                               )
        self.ppo_actor_dist = tfp.distributions.Dirichlet

        self.ppo_critic = self.__build_critic()
        self.ppo_critic.compile(optimizer=keras.optimizers.Adam(learning_rate=self.alpha_critic,
                                                                # epsilon=1.5*10e-4
                                                                ),
                               )
        self.ppo_critic_loss = tf.keras.losses.MeanSquaredError()

        self.experience_replay_buffer = PPO_ExperienceReplay(max_length=int(kwargs['experience_replay_buffer_length']) if 'experience_replay_buffer_length' in kwargs else int(10*1440), #bigger is smoother, but might require a bigger replay buffer
                                                            action_types=self.actions,
                                                             multivariate=True,
                                                            )

        # Buffers we need for logging stuff before putting into the PPo Memory
        self.actions_buffer = {}
        # self.pi_history = {}
        self.log_prob_buffer = {}
        self.value_buffer = {}
        self.state_buffer = {}

        #logs we need for plotting
        self.rewards_history = []
        self.value_history = []
        self.actions_history = {}
        self.pdf_history = {}
        for action in self.actions:
            self.actions_history[action] = []
            # self.pdf_history[action] = {}
            # for param in ['loc', 'scale']:
            #     self.pdf_history[action][param] = []

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

    def __build_actor(self):
        num_inputs = 2 if 'storage' not in self.actions else 3
        num_hidden = 32 if 'storage' not in self.actions else 64
        num_hidden_layers = 2 #lets see how far we get with this first
        num_actions = len(self.actions)

        initializer = tf.keras.initializers.HeNormal()

        inputs = layers.Input(shape=(num_inputs,), name='Actor_Input')

        internal_signal = inputs
        for hidden_layer_number in range(num_hidden_layers): #hidden layers
            internal_signal = layers.Dense(num_hidden,
                                           activation="elu",
                                           # bias_initializer=initializer,
                                           kernel_initializer=initializer,
                                           name='Actor_Hidden' + str(hidden_layer_number))(internal_signal)

        concentration = layers.Dense(num_actions,
                                  activation=None,
                                  kernel_initializer=initializer,
                                  name='concentration')(internal_signal)
        concentration = tf.where(tf.math.greater(concentration,1.0),  #essentially just huber function it so its bigger than 0
                                 tf.abs(concentration),
                                 tf.square(concentration))
        concentration = concentration + 1e-10 #so we prevent it from being 0

        return keras.Model(inputs=inputs, outputs=concentration)

    def __build_critic(self):
        num_inputs = 2 if 'storage' not in self.actions else 3
        num_hidden = 32 if 'storage' not in self.actions else 64
        num_hidden_layers = 2 #lets see how far we get with this first

        initializer = tf.keras.initializers.HeNormal()
        inputs = layers.Input(shape=(num_inputs,), name='Actor_Input')

        internal_signal = inputs
        for hidden_layer_number in range(num_hidden_layers): #hidden layers
            internal_signal = layers.Dense(num_hidden,
                                           activation="elu",
                                           kernel_initializer=initializer,
                                           name='Actor_Hidden' + str(hidden_layer_number))(internal_signal)

        value = layers.Dense(1,
                             activation=None,
                             kernel_initializer=initializer,
                             name='ValueHead')(internal_signal)

        return keras.Model(inputs=inputs, outputs=value)

    def anneal(self, parameter:str, adjustment, mode:str='multiply', limit=None):
        if not hasattr(self, parameter):
            return False

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
            return
        # align reward with action timing
        # in the current market setup the reward is for actions taken 3 steps ago
        # if self._rewards.type == 'net_profit':
        reward_time_offset = current_round[1] - next_settle[1] - round_duration
        reward_timestamp = current_round[1] + reward_time_offset

        await self.metrics.track('rewards', reward)
        self.rewards_history.append(reward)

        if reward_timestamp in self.state_buffer and reward_timestamp in self.actions_buffer:  # we found matching ones, buffer and pop
            #ToDO: are we supposed to log the log_probs or the normal probs??
            self.experience_replay_buffer.add_entry(states=self.state_buffer[reward_timestamp],
                                                    actions_taken=self.actions_buffer[reward_timestamp],
                                                    log_probs=self.log_prob_buffer[reward_timestamp],
                                                    values=self.value_buffer[reward_timestamp],
                                                    rewards=reward,
                                                    episode=self.gen)

            self.actions_buffer.pop(reward_timestamp) #ToDo: check if we can pop into the above function, would look nicer
            self.log_prob_buffer.pop(reward_timestamp)
            # self.pi_history.pop(reward_timestamp)
            self.value_buffer.pop(reward_timestamp)
            self.state_buffer.pop(reward_timestamp)

            if self.experience_replay_buffer.should_we_learn():
                advantage_calulated = await self.experience_replay_buffer.calculate_advantage(gamma=self.gamma,
                                                                                              gae_lambda=self.gae_lambda,
                                                                                              normalize=self.normalize_advantages,
                                                                                              entropy_reg=self.entropy_reg
                                                                                              )  # ToDo: check once more on a different immplementation if this is right
                buffer_indexed = await self.experience_replay_buffer.generate_availale_indices()  # so we can caluclate the batches faster
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, func=self.train_RL_agent)

    def train_RL_agent(self):

        print('started learning')

        early_stop_critic = False #ToDo: implement these
        early_stop_actor = False
        max_train_steps = self.train_step + self.max_train_steps

        critic_stopper = EarlyStopper(patience=self.critic_patience)

        while self.train_step <= max_train_steps and not (early_stop_actor and early_stop_critic):

            batch = self.experience_replay_buffer.fetch_batch(batchsize=self.batch_size) #to see the batch data structure check this method
            states = tf.convert_to_tensor(batch['states'], dtype=tf.float32)
            advantages = tf.convert_to_tensor(batch['advantages'], dtype=tf.float32)


            if not early_stop_critic:
                with tf.GradientTape() as tape_critic:
                #calculate critic loss and backpropagate

                    critic_Vs = self.ppo_critic(states)
                    losses_critic = self.ppo_critic_loss(critic_Vs, tf.convert_to_tensor(batch['Return'], dtype=tf.float32))

                    # calculate the stopping crtierions
                    early_stop_critic, self.ppo_critic = critic_stopper.check_iteration(losses_critic.numpy(),
                                                                                    self.ppo_critic)
                    # stop or update
                    if early_stop_critic:
                        print('stopping training the critic early, after ', self.train_step + self.max_train_steps - max_train_steps)
                    else:
                        critic_vars = self.ppo_critic.trainable_variables
                        critic_grads = tape_critic.gradient(losses_critic, critic_vars)
                        self.ppo_critic.optimizer.apply_gradients(zip(critic_grads, critic_vars))

                        with self.summary_writer.as_default():
                            tf.summary.scalar('critic_loss', losses_critic, step=self.train_step)

            if not early_stop_actor:
                log_probs_old = tf.convert_to_tensor(batch['log_probs'])
                a_taken = tf.convert_to_tensor(batch['actions_taken'])

                with tf.GradientTape() as tape_actor:
                    pi_batch_new = self.ppo_actor(states)

                    dist = self.ppo_actor_dist(concentration=pi_batch_new, #ToDO: make sure this works with batch dimension etc!
                                              validate_args=False,
                                               allow_nan_stats=False,
                                               force_probs_to_zero_outside_support=False,
                                               )

                    log_probs_new = dist.log_prob(a_taken)
                    #This is how baselines does it

                    ratio = tf.exp(log_probs_new - log_probs_old)  # pi(a|s) / pi_old(a|s)
                    # entropy = -dist.entropy()
                    entropy = -log_probs_new
                    soft_advantages = advantages + self.entropy_reg * entropy

                    clipped_ratio = tf.clip_by_value(ratio, 1-self.policy_clip, 1+self.policy_clip)
                    weighted_ratio = clipped_ratio * soft_advantages
                    loss_actor = -tf.math.minimum(ratio*soft_advantages, weighted_ratio)
                    loss_actor = tf.math.reduce_mean(loss_actor)
                    loss_actor = tf.math.maximum(loss_actor, -500)


                    # PPO early stopping as implemented in baselines
                    approx_kl = tf.math.reduce_mean(log_probs_old - log_probs_new)

                    # collect entropy because why not. If this keeps growing we might have a too small memory and too smal batchsize
                    entropy = tf.reduce_mean(-log_probs_new)

                    # early stopping condition or keep training, consider having this a running avg of 5 or sth?
                    if tf.math.reduce_mean(approx_kl).numpy() > 1.5 * self.kl_stop:
                        early_stop_actor = True
                        print('stopping actor training due to exceeding KL-divergence tolerance with approx KL of', approx_kl,' after ' , self.train_step + self.max_train_steps - max_train_steps)
                    else:
                        # Backpropagation

                        actor_vars = self.ppo_actor.trainable_variables

                        actor_grads = tape_actor.gradient(loss_actor, actor_vars)
                        sum_grads = tf.reduce_sum(actor_grads[0]).numpy()
                        if np.isnan(sum_grads):
                            print("AAAAAAAA!!")
                        self.ppo_actor.optimizer.apply_gradients(zip(actor_grads, actor_vars))

                        with self.summary_writer.as_default():
                            tf.summary.scalar('actor_loss', loss_actor, step=self.train_step)
                            tf.summary.scalar('approx_KL', approx_kl, step=self.train_step)
                            tf.summary.scalar('entropy', entropy, step=self.train_step)

            self.train_step = self.train_step + 1

        #clear the buffer after we learned it
        self.experience_replay_buffer.clear_buffer()

    async def __sample_pi(self, pi_actions): #ToDo: check how this behaves for batches of actions and for single actions, we want this to be consisten!

        dist = self.ppo_actor_dist(concentration=pi_actions,
                                   validate_args=False,
                                   force_probs_to_zero_outside_support=True,
                                   allow_nan_stats=False)

        a_dist = dist.sample(1)
       #  a_dist = tf.math.minimum(tf.math.maximum(a_dist, 1e-10), 1.0-1e-10)
        a_dist = tf.math.maximum(a_dist, 1e-10)
        log_prob = dist.log_prob(a_dist)

        log_prob = tf.squeeze(log_prob)
        log_prob = log_prob.numpy().tolist()
        if np.isnan(log_prob):
            print('AAAAA')

        a_dist = tf.squeeze(a_dist)
        a_dist = a_dist.numpy().tolist()
        if np.isnan(np.sum(a_dist)):
            print('AAAAA1')

        a_scaled = {}
        keys = list(self.actions.keys())
        for action_index in range(len(keys)):
            a = a_dist[action_index]
            min = self.actions[keys[action_index]]['min']
            max = self.actions[keys[action_index]]['max']
            a = min + (a * (max - min))
            a_scaled[keys[action_index]] = a

        return a_scaled, log_prob, a_dist

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
                 float(next_generation/17),
                 float(next_load/17)]

        if 'storage' in self.__participant: #ToDo: Check if this is the right way to acess SOC
            storage_schedule = await self.__participant['storage']['check_schedule'](next_settle)
            soc = storage_schedule[next_settle]['projected_soc_end']
            state.append(soc)

        state = np.array(state)
        pi_actions = self.ppo_actor(tf.expand_dims(state, axis=0))
        if np.isnan(np.sum(pi_actions.numpy())):
             print("AAAAA3")


        value = self.ppo_critic(tf.expand_dims(state, axis=0))
        value = tf.squeeze(value, axis=0).numpy().tolist()[0]
        taken_action, log_prob, dist_action = await self.__sample_pi(pi_actions)

        # lets log the stuff needed for the replay buffer
        self.state_buffer[current_round[1]] = state
        self.actions_buffer[current_round[1]] = dist_action
        self.log_prob_buffer[current_round[1]] = log_prob
        self.value_buffer[current_round[1]] = value
        self.value_history.append(value)
        # for action in self.actions:
        #     for param in pi_actions:
        #         self.pdf_history[action][param].append(pi_actions[param])

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


        price = taken_action['price']
        price = round(price, 4)
        quantity = taken_action['quantity']
        quantity = int(quantity)

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
        print(self.__participant['id'], 'episode reward:', episode_G)

        with self.summary_writer.as_default():
            tf.summary.scalar('Return' , episode_G, step= self.gen)
            tf.summary.histogram('Rewards during Episode', self.rewards_history, step=self.gen)
            tf.summary.histogram('Values',
                                 self.value_history,
                                 step=self.gen)

            for action in self.actions:
                tf.summary.histogram(action, self.actions_history[action], step=self.gen)


            for layer in self.ppo_critic.layers:
                if layer.name not in ['Input']:
                    for weights in layer.weights:
                        tf.summary.histogram(weights.name, weights.numpy(), step=self.gen)

            for layer in self.ppo_actor.layers:
                if layer.name not in ['Input']:
                    for weights in layer.weights:
                        tf.summary.histogram(weights.name, weights.numpy(), step=self.gen)


        self.gen = self.gen + 1

    async def reset(self, **kwargs):
        self.state_buffer.clear()
        self.value_buffer.clear()
        self.actions_buffer.clear()
        self.log_prob_buffer.clear()

        self.rewards_history.clear()
        self.value_history.clear()
        for action in self.actions:
            self.actions_history[action].clear()
        #     for param in ['loc', 'scale']:
        #         self.pdf_history[action][param].clear()

        return True