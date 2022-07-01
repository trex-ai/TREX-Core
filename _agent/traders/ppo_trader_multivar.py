
import asyncio
import importlib
import os
import random
from collections import OrderedDict

import numpy as np
from _agent._utils.metrics import Metrics
from _utils import utils
from _utils.drl_utils import robust_argmax
from _utils.drl_utils import PPO_ExperienceReplay, EarlyStopper, huber, tb_plotter,build_actor_critic_models
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

def build_multivar(concentrations, dist, actions):
    # independent_dists = []
    # for action in args:
    #     c0 = args[action]['c0']
    #     c1 = args[action]['c1']
    #     dist_action = dist(c0, c1)
    args = {}
    c0s, c1s  = tf.split(concentrations, 2, -1)
    c0s = tf.split(c0s, len(actions), -1)
    #c0s = tf.concat(c0s, axis=-2)
    c1s = tf.split(c1s, len(actions), -1)
    #c1s = tf.concat(c1s, axis=-2)
    args['concentration0'] = tf.concat(c0s, axis=-1)
    args['concentration1'] = tf.concat(c1s, axis=-1)
    betas = dist(**args)
    # sample = betas.sample(1)
    multivar_dist = tfp.distributions.Independent(betas, reinterpreted_batch_ndims=1)
    # sample_dis = multivar_dist.sample(1)
    return multivar_dist

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

        #ToDo: test if having parameter sharing helps here?

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

        self.warmup_actor = kwargs['warmup_actor']
        self.observations = kwargs['observations']

        self.burn_in = kwargs['burn_in'] if 'burn_in' in kwargs else 0
        self.trajectory_length = kwargs['trajectory_length'] if 'trajectory_length' in kwargs else 1

        self.experience_replay_buffer = PPO_ExperienceReplay(max_length=self.replay_buffer_length,
                                                            action_types=self.actions,
                                                             multivariate=True,
                                                             trajectory_length=self.burn_in+self.trajectory_length
                                                            )
        self.ppo_actor, self.ppo_critic, self.ppo_actor_dist = build_actor_critic_models(num_inputs=len(kwargs['observations']),
                                                                                         hidden_actor=kwargs['actor_hidden'],
                                                                                         actor_type=self.actor_type,
                                                                                         hidden_critic=kwargs['critic_hidden'],
                                                                                         critic_type=self.critic_type,
                                                                                         num_actions=len(self.actions))
        self.distribution_type = 'Beta'
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

        #logs we need for plotting
        self.rewards_history = []
        self.value_history = []
        self.observations_history = []
        self.net_load_history = []
        if self.actor_type == 'GRU':
            self.actor_states_history = []
        if self.critic_type == 'GRU':
            self.critis_states_history = []

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
                                                    rewards=reward,
                                                    episode=self.gen)

            self.actions_buffer.pop(reward_timestamp) #ToDo: check if we can pop into the above function, would look nicer
            self.log_prob_buffer.pop(reward_timestamp)
            self.value_buffer.pop(reward_timestamp)
            self.observations_buffer.pop(reward_timestamp)

            if self.experience_replay_buffer.should_we_learn():
                advantage_calulated = await self.experience_replay_buffer.calculate_advantage(gamma=self.gamma,
                                                                                              gae_lambda=self.gae_lambda,
                                                                                              normalize=self.normalize_advantages,
                                                                                              )
                buffer_indexed = await self.experience_replay_buffer.generate_availale_indices()  # so we can caluclate the batches faster
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, func=self.train_RL_agent)

    def pretrain_actor(self, observations, actor_stopper, max_train_steps):
        with tf.GradientTape() as tape_warmup:
            pi_new = self.ppo_actor(observations)

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
            tb_plotter(data_for_tb, self.summary_writer)

            actor_vars = self.ppo_actor.trainable_variables
            actor_grads = tape_warmup.gradient(losses_warmup, actor_vars)
            self.ppo_actor.optimizer.apply_gradients(zip(actor_grads, actor_vars))
        else:
            print('stopping warmup ', max_train_steps - self.train_step, 'steps early')

        return stop_actor_training

    def train_actor(self, observations, a_taken, log_probs_old, advantages):
        stop_actor_training = False
        with tf.GradientTape() as tape_actor:
            pi_batch = self.ppo_actor(observations)
            dist = build_multivar(pi_batch, self.ppo_actor_dist, self.actions)

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
                self.ppo_actor.optimizer.apply_gradients(zip(actor_grads, actor_vars))

            # log
            data_for_tb = [{'name': 'actor_loss', 'data': loss_actor, 'type': 'scalar', 'step': self.train_step},
                           {'name': 'approx_KLD', 'data': approx_kl, 'type': 'scalar',
                            'step': self.train_step},
                           {'name': 'entropy', 'data': entropy, 'type': 'scalar',
                            'step': self.train_step},
                           {'name': 'early_stop_actor', 'data': stop_actor_training, 'type': 'scalar',
                            'step': self.train_step},
                           ]
            tb_plotter(data_for_tb, self.summary_writer)

        return stop_actor_training

    def train_critic(self, observations, Return,  critic_stopper):
        with tf.GradientTape() as tape_critic:
            # calculate critic loss and backpropagate

            critic_Vs = self.ppo_critic(observations)
            critic_Vs = tf.squeeze(critic_Vs, axis=-1)
            G = tf.convert_to_tensor(Return, dtype=tf.float32)
            losses_critic = self.ppo_critic_loss(critic_Vs, G)

            # calculate the stopping crtierions
            if self.use_early_stop_critic:
                stop_critic_training, self.ppo_critic = critic_stopper.check_iteration(losses_critic.numpy(),
                                                                                    self.ppo_critic)
            # log
            data_for_tb = [{'name': 'critic_loss', 'data': losses_critic, 'type': 'scalar', 'step': self.train_step}]
            tb_plotter(data_for_tb, self.summary_writer)
            # early stop or learn
            if stop_critic_training:
                # print('stopping training the critic early, after ', self.train_step + self.max_train_steps - max_train_steps)
                pass
            else:
                critic_vars = self.ppo_critic.trainable_variables
                critic_grads = tape_critic.gradient(losses_critic, critic_vars)
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

            batch = self.experience_replay_buffer.fetch_batch(batchsize=self.batch_size) #to see the batch data structure check this method
            observations = tf.convert_to_tensor(batch['observations'], dtype=tf.float32)
            Return = tf.convert_to_tensor(batch['Return'], dtype=tf.float32)

            if not stop_critic_training:
                stop_critic_training = self.train_critic(observations, Return, critic_stopper)

            if not stop_actor_training:
                advantages = tf.convert_to_tensor(batch['advantages'], dtype=tf.float32)
                log_probs_old = tf.convert_to_tensor(batch['log_probs'], dtype=tf.float32)
                a_taken = tf.convert_to_tensor(batch['actions_taken'], dtype=tf.float32)
                if not self.warmup_actor:
                    stop_actor_training = self.train_actor(observations=observations,
                                                        a_taken=a_taken,
                                                        log_probs_old=log_probs_old,
                                                        advantages=advantages)
                else:
                    stop_actor_training = self.pretrain_actor(observations, actor_stopper, max_train_steps)

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
        # Generate state (inputs to model):
        # - time(s)
        # - next generation
        # - next load
        # - battery stats (if available)

        current_round = self.__participant['timing']['current_round']
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
        # next_settle_end = utils.timestamp_to_local(next_settle[1], timezone)


        observations_t = []
        if 'generation' in self.observations:
            observations_t.append(next_generation/17)
        if 'load' in self.observations:
            observations_t.append(next_load/17)
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
        if self.actor_type == 'FFNN':
            pi_dict = self.ppo_actor(obs_t)
        elif self.actor_type == 'GRU':
            pi_dict, states_actor_t = self.ppo_actor(obs_t)
            self.actor_states_history.append(states_actor_t)

        if self.critic_type == 'FFNN':
            V_t = self.ppo_critic(obs_t)

        elif self.critic_type == 'GRU':
            V_t, states_critic_t = self.ppo_critic(obs_t)
            self.critis_states_history.append(states_critic_t)

        V_t = tf.squeeze(V_t, axis=0).numpy().tolist()[0]
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
        print(self.__participant['id'], 'episode reward:', episode_G)

        data_for_tb = [{'name':'Return', 'data':episode_G, 'type':'scalar', 'step':self.gen},
                       {'name': 'Episode Rewards', 'data': self.rewards_history, 'type': 'histogram', 'step':self.gen},
                       {'name':'Values', 'data':self.value_history, 'type':'histogram', 'step':self.gen}]
        for action in self.actions:
            data_for_tb.append({'name':action, 'data':self.actions_history[action], 'type':'histogram', 'step':self.gen})


        #day_length = 8
        #socs = np.array(self.observations_history)[:,-1]*100
        #data_for_tb.append({'name': 'SoC_during_day', 'data': socs, 'type': 'pseudo3D', 'step':self.gen, 'buckets': day_length})

        #net_load_history = self.net_load_history - np.amin(self.net_load_history)
        #data_for_tb.append(
        #    {'name': 'Effective_Ned_load_during_day', 'data': net_load_history, 'type': 'pseudo3D', 'step': self.gen, 'buckets': day_length})


        tb_plotter(data_for_tb, self.summary_writer)


        self.gen = self.gen + 1

    async def reset(self, **kwargs):
        self.observations_buffer.clear()
        self.value_buffer.clear()
        self.actions_buffer.clear()
        self.log_prob_buffer.clear()


        # states = np.array(self.observations_history)
        # for row in range(states.shape[-1]):
        #     plt.plot(states[:,row])
        #     plt.show()

        self.rewards_history.clear()
        self.value_history.clear()
        self.observations_history.clear()
        self.net_load_history.clear()
        for action in self.actions:
            self.actions_history[action].clear()
        #     for param in ['loc', 'scale']:
        #         self.pdf_history[action][param].clear()

        return True