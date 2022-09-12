
import asyncio
import importlib
import os
import random
from collections import OrderedDict

import numpy as np
from _agent._utils.metrics import Metrics
from _utils import utils
from _utils.drl_utils import robust_argmax
from _utils.drl_utils import PPO_ExperienceReplay, EarlyStopper, huber, tb_plotter,build_actor_critic_models, build_multivar, assemble_subdict_batch
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
import tensorflow_probability as tfp
tf.get_logger().setLevel(3)


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
        experiment_path = os.path.join(cwd, self.study_name)
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

        # Hyperparameters
        self.alpha_critic = kwargs['alpha_critic']
        self.alpha_actor = kwargs['alpha_actor']
        self.actor_type = kwargs['actor_type']

        self.batch_size = kwargs['batch_size'] #bigger is smoother, but might require a bigger replay buffer
        self.policy_clip = kwargs['policy_clip']
        self.kl_stop = kwargs['kl_stop'] #according to baselines tends to be bewteen 0.01 and 0.05
        self.entropy_reg = kwargs['entropy_reg']
        self.gamma = kwargs['gamma']
        self.gae_lambda = kwargs['gae_lambda']
        self.normalize_advantages = kwargs['normalize_advantages']
        self.use_early_stop_actor = kwargs['use_early_stop_actor']

        self.critic_patience = kwargs['critic_patience']
        self.use_early_stop_critic = kwargs['use_early_stop_critic']
        self.critic_type = kwargs['critic_type']

        self.max_train_steps = kwargs['max_train_steps']
        self.replay_buffer_length = kwargs['experience_replay_buffer_length']
        self.g_grad_norm = kwargs['g_grad_norm']

        self.warmup_actor = kwargs['warmup_actor']
        self.observations = kwargs['observations']

        self.burn_in = kwargs['burn_in'] if 'burn_in' in kwargs else 0
        self.trajectory_length = kwargs['trajectory_length'] if 'trajectory_length' in kwargs else 1

        self.experience_replay_buffer = PPO_ExperienceReplay(max_length=self.replay_buffer_length,
                                                            action_types=self.actions,
                                                             multivariate=True,
                                                             trajectory_length=self.burn_in+self.trajectory_length)


        actor_dict, critic_dict = build_actor_critic_models(num_inputs=len(kwargs['observations']),
                                                                                         hidden_actor=kwargs['actor_hidden'],
                                                                                         actor_type=self.actor_type,
                                                                                         hidden_critic=kwargs['actor_hidden'],
                                                                                         critic_type=self.critic_type,
                                                                                         num_actions=len(self.actions))

        self.ppo_actor = actor_dict['model']
        self.ppo_actor_dist = actor_dict['distribution']
        if self.actor_type == 'GRU':
            self.actor_states_dummy = actor_dict['initial_states_dummy']

        self.ppo_critic = critic_dict['model']
        if self.critic_type == 'GRU':
            self.critic_states_dummy = critic_dict['initial_states_dummy']

        self.ppo_actor.compile(optimizer=k.optimizers.Adam(learning_rate=self.alpha_actor,),)
        if self.warmup_actor:
            self.ppo_actor_warmup_loss = k.losses.MeanSquaredError()

        self.ppo_critic.compile(optimizer=k.optimizers.Adam(learning_rate=self.alpha_critic,),)
        self.ppo_critic_loss = k.losses.MeanSquaredError()

        # Buffers we need for logging stuff before putting into the PPo Memory
        self.actions_buffer = {}
        # self.pi_history = {}
        self.log_prob_buffer = {}
        self.value_buffer = {}
        self.observations_buffer = {}
        if self.actor_type == 'GRU':
            self.actor_states_buffer = {}
        if self.critic_type == 'GRU':
            self.critic_states_buffer = {}

        #logs we need for plotting
        self.rewards_history = []
        self.value_history = []
        self.observations_history = []
        self.net_load_history = []

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
        if reward_timestamp in self.observations_buffer and reward_timestamp in self.actions_buffer:  # we found matching ones, buffer and pop

            self.experience_replay_buffer.add_entry(observations=self.observations_buffer[reward_timestamp],
                                                    actions_taken=self.actions_buffer[reward_timestamp],
                                                    log_probs=self.log_prob_buffer[reward_timestamp],
                                                    values=self.value_buffer[reward_timestamp],
                                                    critic_states= self.critic_states_buffer[reward_timestamp] if self.critic_type == 'GRU' else None,
                                                    actor_states = self.actor_states_buffer[reward_timestamp] if self.actor_type == 'GRU' else None,
                                                    rewards=reward,
                                                    episode=self.gen)

            self.actions_buffer.pop(reward_timestamp) #ToDo: check if we can pop into the above function, would look nicer
            self.log_prob_buffer.pop(reward_timestamp)
            self.value_buffer.pop(reward_timestamp)
            self.observations_buffer.pop(reward_timestamp)
            if self.actor_type == 'GRU':
                self.actor_states_buffer.pop(reward_timestamp)
            if self.critic_type == 'GRU':
                self.critic_states_buffer.pop(reward_timestamp)

            if self.experience_replay_buffer.should_we_learn():
                advantage_calulated = await self.experience_replay_buffer.calculate_advantage(gamma=self.gamma,
                                                                                              gae_lambda=self.gae_lambda,
                                                                                              normalize=self.normalize_advantages,
                                                                                              )
                buffer_indexed = await self.experience_replay_buffer.generate_availale_indices()  # so we can caluclate the batches faster
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, func=self.train_RL_agent)

    def pretrain_actor(self, actor_inputs, actor_stopper, max_train_steps):

        with tf.GradientTape() as tape_warmup:
            actor_outputs = self.ppo_actor(actor_inputs)
            pi_new = actor_outputs.pop('pi')
            if self.burn_in > 0:
                pi_new = pi_new[:, self.burn_in:, :]

            targets = tf.ones(shape=pi_new.shape)
            losses_warmup = self.ppo_actor_warmup_loss(targets, pi_new)

            losses_warmup = tf.reduce_mean(losses_warmup)

        # calculate the stopping crtierions
        if self.use_early_stop_critic:
            stop_actor_training, self.ppo_actor = actor_stopper.check_iteration(losses_warmup.numpy(),
                                                                             self.ppo_actor)

        # early stop or learn
        if not stop_actor_training:
            data_for_tb = [{'name': 'actor_warmup_loss',
                            'data': losses_warmup,
                            'type': 'scalar',
                            'step': self.train_step}]

            # loop = asyncio.get_running_loop()
            # await loop.run_in_executor(None, tb_plotter, [data_for_tb, self.summary_writer])
            tb_plotter(data_for_tb, self.summary_writer)

            actor_vars = self.ppo_actor.trainable_variables
            actor_grads = tape_warmup.gradient(losses_warmup, actor_vars)
            self.ppo_actor.optimizer.apply_gradients(zip(actor_grads, actor_vars))
        else:
            # print('stopping warmup ', max_train_steps - self.train_step, 'steps early')
            pass

        return stop_actor_training

    def train_actor(self, actor_inputs, a_taken, log_probs_old, advantages):
        stop_actor_training = False
        with tf.GradientTape() as tape_actor:
            actor_outputs = self.ppo_actor(actor_inputs)
            pi = actor_outputs.pop('pi')
            if self.burn_in > 0:
                pi = pi[:, self.burn_in:, :]
                a_taken = a_taken[:, self.burn_in:, :]
                log_probs_old = log_probs_old[:, self.burn_in:, :]
                advantages = advantages[:, self.burn_in:]

            dist = build_multivar(pi, self.ppo_actor_dist, self.actions)

            log_probs_new = dist.log_prob(a_taken)
            # check = tf.reduce_sum(log_probs_new).numpy()
            # if np.isnan(check) or np.isinf(check):
            #     probs = dist.prob(a_taken)
            #     print('shit')
            # This is how baselines does it
            log_probs_new = tf.squeeze(log_probs_new)
            log_probs_old = tf.squeeze(log_probs_old)

            ratio = tf.exp(log_probs_new - log_probs_old)  # pi(a|s) / pi_old(a|s)
            # entropy = -dist.entropy()
            entropy = -log_probs_new
            soft_advantages = (1.0 - self.entropy_reg) * advantages + self.entropy_reg * entropy

            clipped_ratio = tf.clip_by_value(ratio, 1 - self.policy_clip, 1 + self.policy_clip)
            weighted_ratio = clipped_ratio * soft_advantages
            loss_actor = -tf.math.minimum(ratio * soft_advantages, weighted_ratio)
            # loss_actor = tf.clip_by_value(loss_actor, clip_value_min=-100, clip_value_max=100)
            loss_actor = tf.math.reduce_mean(loss_actor)

            # PPO early stopping as implemented in baselines
            approx_kl = tf.math.reduce_mean(log_probs_old - log_probs_new)

            # collect entropy because why not. If this keeps growing we might have a too small memory and too smal batchsize
            entropy = tf.reduce_mean(-log_probs_new)

            # early stopping condition or keep training, consider having this a running avg of 5 or sth?
            if self.use_early_stop_actor:
                if tf.math.reduce_mean(approx_kl).numpy() > 1.5 * self.kl_stop:
                    stop_actor_training = True
                    # print('stopping actor training due to exceeding KL-divergence tolerance with approx KL of', approx_kl,' after ' , self.train_step + self.max_train_steps - max_train_steps)

            if not stop_actor_training:
                # Backpropagation
                actor_vars = self.ppo_actor.trainable_variables
                actor_grads = tape_actor.gradient(loss_actor, actor_vars)
                actor_grads, _ = tf.clip_by_global_norm(actor_grads, self.g_grad_norm)
                self.ppo_actor.optimizer.apply_gradients(zip(actor_grads, actor_vars))

            # log
            data_for_tb = [{'name': 'actor_loss', 'data': loss_actor, 'type': 'scalar', 'step': self.train_step}, #Main loss, if too spiky we want to see where it comes from
                           {'name': 'ratio', 'data': tf.reduce_mean(ratio), 'type': 'scalar', 'step': self.train_step}, #Ratio of old and new policy probabilities .... Loss component
                           {'name': 'weighted_ratio', 'data': tf.reduce_mean(weighted_ratio), 'type': 'scalar', 'step': self.train_step}, #Loss componens
                           {'name': 'ratio x SoftAdv', 'data': tf.reduce_mean(ratio * soft_advantages), 'type': 'scalar', 'step': self.train_step}, #Loss componens
                           {'name': 'approx_KLD', 'data': approx_kl, 'type': 'scalar', 'step': self.train_step}, #Distance pseudometric between old and new policy, we want this to decrease over training as this would indicate convergence
                           {'name': 'entropy', 'data': entropy, 'type': 'scalar', 'step': self.train_step}, #Randomness of policy, we want the differential entropy to keep dropping slowly over the course of training
                           {'name': 'early_stop_actor', 'data': stop_actor_training, 'type': 'scalar',
                            'step': self.train_step},
                           ]

            # loop = asyncio.get_running_loop()
            # await loop.run_in_executor(None, tb_plotter, [data_for_tb, self.summary_writer])
            tb_plotter(data_for_tb, self.summary_writer)

        return stop_actor_training

    def train_critic(self, critic_inputs, returns,  critic_stopper):
        with tf.GradientTape() as tape_critic:
            # calculate critic loss and backpropagate

            critic_outputs = self.ppo_critic(critic_inputs)
            Vs = critic_outputs.pop('value')
            Vs = tf.squeeze(Vs, axis=-1)

            if self.burn_in > 0: #discard unwanted stages
                Vs = Vs[:, self.burn_in:]
                returns = returns[:, self.burn_in:]
            losses_critic = self.ppo_critic_loss(Vs, returns)

            # calculate the stopping crtierions
            if self.use_early_stop_critic:
                stop_critic_training, self.ppo_critic = critic_stopper.check_iteration(losses_critic.numpy(),
                                                                                    self.ppo_critic)
            # log
            data_for_tb = [{'name': 'critic_loss', 'data': losses_critic, 'type': 'scalar', 'step': self.train_step},]
            tb_plotter(data_for_tb, self.summary_writer)
            # early stop or learn
            if stop_critic_training:
                # print('stopping training the critic early, after ', self.train_step + self.max_train_steps - max_train_steps)
                pass
            else:
                critic_vars = self.ppo_critic.trainable_variables
                critic_grads = tape_critic.gradient(losses_critic, critic_vars)
                critic_grads, _ = tf.clip_by_global_norm(critic_grads, self.g_grad_norm)
                self.ppo_critic.optimizer.apply_gradients(zip(critic_grads, critic_vars))

        return stop_critic_training

    def train_RL_agent(self):

        stop_critic_training = False
        stop_actor_training = False
        max_train_steps = self.train_step + self.max_train_steps

        critic_stopper = EarlyStopper(patience=self.critic_patience)
        if self.warmup_actor:
            actor_stopper = EarlyStopper(patience=self.critic_patience)

        while self.train_step <= max_train_steps and not (stop_actor_training and stop_critic_training):
            keys_to_fetch = ['returns', 'observations', 'advantages', 'actions_taken', 'log_probs', 'critic_states', 'actor_states']
            batch = self.experience_replay_buffer.fetch_batch(batchsize=self.batch_size, keys=keys_to_fetch) #to see the batch data structure check this method
            observations = tf.convert_to_tensor(batch['observations'], dtype=tf.float32)


            if not stop_critic_training:
                # manage the inputs
                returns = tf.convert_to_tensor(batch['returns'], dtype=tf.float32)

                critic_inputs = {'observations': observations}
                if self.critic_type == 'GRU':                                                       # this still seems somewhat unclean
                    critic_states = assemble_subdict_batch(batch['critic_states'])
                    for key in self.critic_states_dummy:
                        states = tf.convert_to_tensor(critic_states[key])
                        states = tf.squeeze(states, axis=1) #ToDo: this should not be necessary really ....
                        critic_inputs[key] = states

                stop_critic_training = self.train_critic(critic_inputs, returns, critic_stopper)

            if not stop_actor_training:

                actor_inputs = {'observations': observations}
                if self.actor_type == 'GRU':
                    actor_states = assemble_subdict_batch(batch['actor_states'])
                    for key in self.actor_states_dummy:
                        states = tf.convert_to_tensor(actor_states[key])
                        states = tf.squeeze(states, axis=1) #ToDo: this should not be necessary really ....
                        actor_inputs[key] = states

                if not self.warmup_actor:
                    advantages = tf.convert_to_tensor(batch['advantages'], dtype=tf.float32)
                    log_probs_old = tf.convert_to_tensor(batch['log_probs'], dtype=tf.float32)
                    a_taken = tf.convert_to_tensor(batch['actions_taken'], dtype=tf.float32)

                    stop_actor_training = self.train_actor(actor_inputs=actor_inputs,
                                                        a_taken=a_taken,
                                                        log_probs_old=log_probs_old,
                                                        advantages=advantages)
                else:
                    stop_actor_training = self.pretrain_actor(actor_inputs, actor_stopper, max_train_steps)

            self.train_step = self.train_step + 1

        #clear the buffer after we learned it
        self.experience_replay_buffer.clear_buffer()
        self.train_step = max_train_steps #to make sure we log all the algorithms that might be running in parallel at the same scales

    async def __sample_pi(self, pi_dict):
        dist = build_multivar(pi_dict, self.ppo_actor_dist, self.actions)

        a_dist = dist.sample(1)
        a_dist = tf.clip_by_value(a_dist, 1e-8, 0.999999)
        carinality = len(a_dist.get_shape())
        to_be_reduced = np.arange(carinality - 1, dtype=int).tolist()
        a_dist = tf.squeeze(a_dist, axis=to_be_reduced)
        a_dist = a_dist.numpy().tolist()

        log_prob = dist.log_prob(a_dist)
        carinality = len(log_prob.get_shape())
        to_be_reduced = np.arange(carinality-1, dtype=int).tolist()
        log_prob = tf.squeeze(log_prob, axis=to_be_reduced)
        log_prob = log_prob.numpy().tolist()

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

        current_round = self.__participant['timing']['current_round']
        previous_round = self.__participant['timing']['last_round']
        next_settle = self.__participant['timing']['next_settle']
        next_generation, next_load = await self.__participant['read_profile'](next_settle)
        self.net_load = next_load - next_generation

        # if 'quantity' in self.actions: #pseudosmart quantities because so far we havent figure out how to manage the full action space...
        #     if self.net_load > 0:
        #         self.actions['quantity']['min'] = 0.0
        #         self.actions['quantity']['max'] = self.net_load
        #     else:
        #         self.actions['quantity']['min'] = self.net_load
        #         self.actions['quantity']['max'] = 0.0

        timezone = self.__participant['timing']['timezone']
        current_round_end = utils.timestamp_to_local(current_round[1], timezone)

        observations_t = []
        if not hasattr(self, 'profile_stats'):
            self.profile_stats = await self.__participant['get_profile_stats']()

        if self.profile_stats:
            if 'generation' in self.observations:
                avg_generation = self.profile_stats['avg_generation']
                stddev_generation = self.profile_stats['stddev_generation']
                z_next_generation = (next_generation - avg_generation) / stddev_generation
                observations_t.append(z_next_generation)
            if 'load' in self.observations:
                avg_load = self.profile_stats['avg_consumption']
                stddev_load = self.profile_stats['stddev_consumption']
                z_next_load = (next_load - avg_load) / stddev_load
                observations_t.append(z_next_load)
        else:
            if 'generation' in self.observations:
                observations_t.append(next_generation)
            if 'load' in self.observations:
                observations_t.append(next_load)

        if 'time_sin' or 'time_cos' in self.observations:
            minutes = int(current_round[0]/60)
            if 'time_sin' in self.observations:
                observations_t.append(np.sin(2*np.pi*minutes/24)) #ToDo: ATM THIS IS ONLY FOR THE FAKE 24H synth profile!!
            if 'time_cos' in self.observations:
                observations_t.append(np.cos(2 * np.pi * minutes / 24))  # ToDo: ATM THIS IS ONLY FOR THE FAKE 24H synth profile!!


        # print('gen', next_generation, 'load', next_load, 'time', current_round[0])
        if 'soc' in self.observations:
            storage_schedule = await self.__participant['storage']['check_schedule'](current_round)
            soc = storage_schedule[current_round]['projected_soc_end']
            # battery_out_current = storage_schedule[current_round]['energy_scheduled']
            observations_t.append(soc)

        observations_t = np.array(observations_t)
        obs_t = tf.expand_dims(observations_t, axis=0)
        if self.actor_type == 'GRU' and self.critic_type == 'GRU':
            obs_t = tf.expand_dims(obs_t, axis=0)

        model_inputs = {}
        model_inputs['observations'] = obs_t

        if self.actor_type == 'GRU':
            actor_inputs = model_inputs
            if previous_round[1] in self.actor_states_buffer:
                last_states = self.actor_states_buffer[previous_round[1]]
            else:
                last_states = self.actor_states_dummy
            for key in last_states:
                actor_inputs[key] = last_states[key]
            actor_outputs = self.ppo_actor(actor_inputs)
            pi_dict = actor_outputs.pop('pi')
            states_actor_t = actor_outputs
            self.actor_states_buffer[current_round[1]] = states_actor_t
        else:
            actor_outputs = self.ppo_actor(model_inputs)
            pi_dict = actor_outputs.pop('pi')

        if self.critic_type == 'GRU':
            critic_inputs = model_inputs
            if previous_round[1] in self.critic_states_buffer:
                last_states = self.critic_states_buffer[previous_round[1]]
            else:
                last_states = self.critic_states_dummy
            for key in last_states:
                critic_inputs[key] = last_states[key]
            critic_outputs = self.ppo_critic(critic_inputs)
            V_t = critic_outputs.pop('value')
            states_critic_t = critic_outputs
            self.critic_states_buffer[current_round[1]] = states_critic_t
        else:
            critic_outputs = self.ppo_critic(model_inputs)
            V_t = critic_outputs.pop('value')

        # log
        #ToDo: add calculation for explained variance once Lab is back online, see https://github.com/ray-project/ray/blob/7f03368fc0f56fee478e9ac15576b626fb1103a9/rllib/utils/tf_utils.py
        data_for_tb = [{'name': 'actor_loss', 'data': V_t, 'type': 'scalar', 'step': self.total_step}, #This we want to increase during training, as that would indicate our agent thinks is going to do better
                       {'name': 'obs_mean', 'data': np.mean(observations_t), 'type': 'scalar', 'step': self.total_step}, #These should be consistent-ish wrt to each other and not super spiky (think orders of magnitude)
                       {'name': 'obs_median', 'data': np.median(observations_t), 'type': 'scalar', 'step': self.total_step},
                      ]
        tb_plotter(data_for_tb, self.summary_writer)

        V_t = tf.squeeze(V_t).numpy().tolist()
        taken_action, log_prob, dist_action = await self.__sample_pi(pi_dict)

        # lets log the stuff needed for the replay buffer
        self.observations_buffer[current_round[1]] = observations_t
        self.actions_buffer[current_round[1]] = dist_action
        self.log_prob_buffer[current_round[1]] = log_prob
        self.value_buffer[current_round[1]] = V_t
        self.value_history.append(V_t)

        self.observations_history.append(observations_t)

        current_generation, current_load = await self.__participant['read_profile'](current_round)
        if 'storage' in self.__participant:
            net_load_current = current_load - current_generation + storage_schedule[current_round]['energy_scheduled']
        else:
            net_load_current = current_load - current_generation
        self.net_load_history.append(net_load_current)

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
            price = 0.111

        if 'quantity' in taken_action:
            quantity = int(taken_action['quantity'])
        else:
            quantity = self.net_load

        if 'storage' in taken_action:
            storage = int(taken_action['storage'])

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
                       {'name':'Values', 'data':self.value_history, 'type':'histogram', 'step':self.gen}]
        for action in self.actions:
            data_for_tb.append({'name':action, 'data':self.actions_history[action], 'type':'histogram', 'step':self.gen})

        day_length = 24 #ToDo: find a way to make this auto adjust....
        socs = np.array(self.observations_history)[:,-1]*100
        data_for_tb.append({'name': 'SoC_during_day', 'data': socs, 'type': 'pseudo3D', 'step': self.gen, 'buckets': day_length})

        net_load_history = self.net_load_history - np.amin(self.net_load_history)
        data_for_tb.append(
           {'name': 'Effective_Ned_load_during_day', 'data': net_load_history, 'type': 'pseudo3D', 'step': self.gen, 'buckets': day_length})

        # loop = asyncio.get_running_loop()
        # await loop.run_in_executor(None, tb_plotter, [data_for_tb, self.summary_writer])
        tb_plotter(data_for_tb, self.summary_writer)

        self.gen = self.gen + 1

    async def reset(self, **kwargs):
        self.observations_buffer.clear()
        self.value_buffer.clear()
        self.actions_buffer.clear()
        self.log_prob_buffer.clear()

        self.rewards_history.clear()
        self.value_history.clear()
        self.observations_history.clear()
        self.net_load_history.clear()
        for action in self.actions:
            self.actions_history[action].clear()

        return True