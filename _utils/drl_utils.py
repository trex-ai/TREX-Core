import random
import tensorflow as tf
from random import randint
import numpy as np
from _utils import utils
from collections import OrderedDict, Counter
import itertools
import scipy.signal
from tensorflow import keras as k
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow_probability as tfp

def build_hidden(signal, type='FFNN', num_hidden=32, name='Actor', initializer=k.initializers.HeNormal()):

    if type == 'FFNN':
        signal = k.layers.Dense(num_hidden,
                                         activation="elu",
                                         kernel_initializer=initializer,
                                         name=name)(signal)
        return signal, None
    elif type == 'GRU':
        signal, last_state = k.layers.GRU(num_hidden,
                              # activation="elu",
                              kernel_initializer=initializer,
                              return_sequences=True, return_state=True,
                              name=name)(signal)
        return signal, last_state

    else:
        print('requested layer type (', type, ') not recognized, failed to build ', name)
        return False, False

def build_actor(num_inputs=4, num_actions=3, hidden_actor=[32, 32, 32], actor_type='FFNN'):
    initializer = k.initializers.HeNormal()

    shape = (num_inputs,) if actor_type != 'GRU' else (None, num_inputs,)
    inputs = k.layers.Input(shape=shape, name='Actor_Input')
    internal_signal = inputs

    hidden_layer = 0
    states = []
    for num_hidden_neurons in hidden_actor:
        internal_signal, state = build_hidden(internal_signal,
                                       type=actor_type,
                                       num_hidden=num_hidden_neurons,
                                       name='Actor_hidden_' + str(hidden_layer))
        states.append(state)
        hidden_layer += 1

    concentrations = k.layers.Dense(2 * num_actions,
                                    activation=None,
                                    kernel_initializer=initializer,
                                    name='concentrations')(internal_signal)
    concentrations = huber(concentrations)
    if actor_type == 'GRU':
        actor_model = k.Model(inputs=inputs, outputs=[concentrations, states])
    else:
        actor_model = k.Model(inputs=inputs, outputs=concentrations)

    actor_distrib = tfp.distributions.Beta

    return actor_model, actor_distrib

def build_critic(num_inputs=4, hidden_critic=[32, 32, 32], critic_type='FFNN'):
    initializer = k.initializers.HeNormal()

    shape = (num_inputs,) if critic_type != 'GRU' else (None, num_inputs,)
    inputs = k.layers.Input(shape=shape, name='Critic_Input')

    internal_signal = inputs
    hidden_layer = 0
    states = []
    for num_hidden_neurons in hidden_critic:
        internal_signal, state = build_hidden(internal_signal,
                                       type=critic_type,
                                       num_hidden=num_hidden_neurons,
                                       name='Critic_hidden' + str(hidden_layer))
        hidden_layer += 1
    value = k.layers.Dense(1,
                           activation=None,
                           kernel_initializer=initializer,
                           name='ValueHead')(internal_signal)

    if critic_type == 'GRU':
        critic_model = k.Model(inputs=inputs, outputs=[value, states])
    else:
        critic_model = k.Model(inputs=inputs, outputs=value)

    return critic_model

def build_actor_critic_models(num_inputs=4,
                              hidden_actor=[32, 32, 32],
                              actor_type='FFNN', #['FFNN', 'GRU'] #ToDo
                              hidden_critic=[32,32,32],
                              critic_type='FFNN', #['FFNN', 'GRU'] #ToDo
                              num_actions=4):
    # needs to return a suitable actor ANN, ctor PDF function and critic ANN
    initializer = tf.keras.initializers.HeNormal()
    actor_model, actor_distrib = build_actor(num_inputs=num_inputs,
                              num_actions=num_actions,
                              hidden_actor=hidden_actor,
                              actor_type=actor_type)
    critic_model = build_critic(num_inputs=num_inputs,
                              hidden_critic=hidden_critic,
                              critic_type=critic_type)

    return actor_model, critic_model, actor_distrib

def tb_plotter(data_list, summary_writer):
    with summary_writer.as_default():
        for entry in data_list:
            type = entry['type']
            name = entry['name']
            data = entry['data']
            step = entry['step']
            if type == 'scalar':
                tf.summary.scalar(name, data, step)
            elif type == 'histogram':
                if 'buckets' in entry:
                    buckets = entry['buckets']
                else:
                    buckets = None
                tf.summary.histogram(name, data, step, buckets=buckets)
            elif type == 'pseudo3D':
                if 'buckets' not in entry:
                    print('did not assign periodicity/buckets value,expect break')
                else:
                    periodicity = entry['buckets']

                to_be_discarded = data.shape[-1]%periodicity
                data = data[:-to_be_discarded]
                data = np.reshape(data, (-1,periodicity))
                data = np.average(data, axis=0)
                data = np.squeeze(data)

                pseudo_counts = []
                for t in range(periodicity):
                    count = int(data[t])
                    for _ in range(count):
                        pseudo_counts.append(t)
                tf.summary.histogram(name, pseudo_counts, step, buckets=buckets)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

# this is a function to robustly pick the argmax in a random fashion if we happen to have several identically maximal values
def huber(x, epsilon=1e-10):
    x = tf.where(tf.math.greater(x, 1.0),
                             # essentially just huber function it so its bigger than 0
                             tf.abs(x),
                             tf.square(x))
    if epsilon > 0:
        x = tf.where(tf.math.greater(x, epsilon),
                     # essentially just huber function it so its bigger than 0
                     x,
                     epsilon)
    return x

def tf_shuffle_axis(value, axis=0, seed=None, name=None):
    perm = list(range(tf.rank(value)))
    perm[axis], perm[0] = perm[0], perm[axis]
    value = tf.random.shuffle(tf.transpose(value, perm=perm))
    value = tf.transpose(value, perm=perm)
    return value

async def robust_argmax(tensor):
    max_value = tf.reduce_max(tensor)
    max_value_idxs = tf.where(tf.math.equal(max_value, tf.squeeze(tensor, axis=0)))
    random_max_value_idx = tf.random.shuffle(max_value_idxs)[0]
    return random_max_value_idx

class EarlyStopper:
    def __init__(self, patience=30, tolerance=1e-8):
        self.patience = patience
        self.tolerance = tolerance

        self.best_loss = np.inf
        self.iterations_not_improved = 0

        self.best_model = None

    def check_iteration(self, loss, model):
        stop_early = False

        if loss <= self.best_loss:
            self.best_loss = loss
            self.best_model = model
            self.iterations_not_improved = 0

        else:
            self.iterations_not_improved += 1

            if self.iterations_not_improved >= self.patience:
                stop_early = True
                model = self.best_model

        return stop_early, model

def normalize_buffer_entry(buffer, key):
    array = []
    for episode in buffer.keys():
        episode_array = [step[key] for step in buffer[episode]]
        array.extend(episode_array)

    mean = np.mean(array)
    std = np.std(array)

    for episode in buffer.keys():
        for t in range(len(buffer[episode])):
            buffer[episode][t][key] = (buffer[episode][t][key] - mean) / (std + 1e-10)

    return buffer

class ExperienceReplayBuffer:
    def __init__(self, max_length=1e4, learn_wait=100, n_steps=1):
        self.max_length = max_length
        self.learn_wait = learn_wait
        self.buffer = {} # a dict of lists, each entry is an episode which is itself a list of entries such as below
        self.last_episode = []
        self.n_steps = n_steps #trajectory length
        # each entry on an episode_n_transitions looks like this
        # entry = {   'a': None, #actions
        #            's': None, #states
        #            'r': None, #rewards
        #            'episode': None #episode number
        #            }

    def add_entry(self, actions, states, rewards, episode=0, ts=None):
        entry = {'a': actions,  # actions
                 's': states,  # states
                 'r': rewards,  # rewards
                 'episode': episode,
                 }


        if episode not in self.buffer:
            self.buffer[episode] = []
        self.buffer[episode].append(entry)

        self._crop_buffer()

    def _get_buffer_length(self):
        buffer_length = 0
        for episode in self.buffer:
            buffer_length += len(self.buffer[episode])
        return buffer_length

    def _crop_buffer(self):
        buffer_length = self._get_buffer_length()
        if buffer_length > self.max_length: #make sure its the right length
            difference = buffer_length - self.max_length
            keys = list(self.buffer.keys())
            while difference > 0:
                oldest_episode_length = len(self.buffer[keys[0]])
                removed = min(oldest_episode_length, difference)
                difference -= removed

                if removed == oldest_episode_length:
                    self.buffer.pop(keys[0])
                    del keys[0]
                else:
                    self.buffer[keys[0]] = self.buffer[keys[0]][removed:]


    def clear_buffer(self):
        self.buffer = []

    # def match_episodes(self, candidate_index, trajectory_length=1):
    #     start = self.buffer[candidate_index]['episode']
    #     #ToDO: implement trajectory functionality
    #     trajectory = [self.buffer[candidate_index + i + 1]['episode'] for i  in range(trajectory_length)]
    #     same_episode = np.equal(start, trajectory).tolist()

    def fetch_batch_indices(self, batchsize):
        weightings = []
        buffer_length = 0
        applicable_keys = []
        for key in self.buffer.keys():
            len_episode = len(self.buffer[key])
            if len_episode > self.n_steps:
                weightings.append(len_episode)
                applicable_keys.append(key)
                buffer_length += len_episode
        weightings = [weight/buffer_length for weight in weightings]

        episode_keys = np.random.choice(applicable_keys, batchsize, replace=True, p=weightings).tolist()
        counts = Counter(episode_keys)
        indices = []
        for episode_key in counts:
            if counts[episode_key] >= len(self.buffer[episode_key]):
                replace = False
            else:
                replace = True

            transition_indices = np.random.choice(len(self.buffer[episode_key])-self.n_steps, counts[episode_key], replace=replace).tolist()
            for transition_index in transition_indices:
                indices.append([episode_key, transition_index])

        return indices

    def should_we_learn(self):
        if self._get_buffer_length() > self.learn_wait:
            return True
        else:
            return False

    def fetch_batch(self, batchsize=32, indices=None):

        if indices is None:
            indices =  self.fetch_batch_indices(batchsize)
        rewards = []
        for [sample_episode, trajectory_start] in indices:
            sample_reward_trajectory = [self.buffer[sample_episode][transition]['r'] for transition in range(trajectory_start, trajectory_start+self.n_steps)]
            rewards.append(sample_reward_trajectory)

        batch = {'actions':     [self.buffer[sample_episode][transition]['a'] for [sample_episode, transition] in indices],
                 'rewards':     np.array(rewards), # old 1 step query: [self.buffer[sample][transition]['r'] for [sample, transition] in indices],       #ToDo: this needs to change for N-step Q
                 'states':      np.array([self.buffer[sample_episode][transition]['s'] for [sample_episode, transition] in indices]),
                 'next_states': np.array([self.buffer[sample_episode][transition + self.n_steps]['s'] for [sample_episode, transition] in indices])
                 }
        return batch

#ToDo:
# in order to make this recurrent we'll need to: store states(to initialize)
# have a length argument for the trajectory we extract
class PPO_ExperienceReplay:
    def __init__(self, max_length=1e4, trajectory_length=1, action_types=None, multivariate=True):
        self.max_length = max_length
        self.buffer = {}  # a dict of lists, each entry is an episode which is itself a list of entries such as below
        self.last_episode = []
        self.action_types = action_types
        self.trajectory_length = trajectory_length  # trajectory length
        self.multivariate = multivariate
        # each entry on an episode_n_transitions looks like this
        # entry = {   'as_taken': actions taken
        #            'logprob_as_taken': logprobs of actions taken
        #            'values': values of critic at time #ToDo: probably get rid of this
        #            'observations: #..
        #            'rewards': ...
        #           'episode': #episode
        #            }

    def add_entry(self, actions_taken, log_probs, values, observations, rewards, episode=0):

        entry = {'observations': observations,
                 'logprob': log_probs,
                 'values': values,
                 'a_taken': actions_taken,
                 'rewards': rewards,
                 }

        if episode not in self.buffer:
            self.buffer[episode] = []
        self.buffer[episode].append(entry)

    def clear_buffer(self):
        self.buffer = {}

    async def generate_availale_indices(self):

        #get available indices
        available_indices = []
        for episode in self.buffer:
            for step in range(len(self.buffer[episode]) - self.trajectory_length):
                available_indices.append([episode, step])

        self.available_indices = available_indices
        return True

    def should_we_learn(self):
        buffer_length = 0
        for episode in self.buffer:
            buffer_length += len(self.buffer[episode])

        if buffer_length >= self.max_length:
            return True
        else:
            return False

    async def calculate_advantage(self, gamma=0.99, gae_lambda=0.95, normalize=True, entropy_reg=0.1):         #ToDo: calculate advantage

        for episode in self.buffer:
            V_episode = [step['values'] for step in self.buffer[episode]]
            V_pseudo_terminal = V_episode[-1]

            r_episode = [step['rewards'] for step in self.buffer[episode]]
            r_episode.append(V_pseudo_terminal)
            r_episode = np.array(r_episode)

            G_episode = discount_cumsum(r_episode, gamma)[:-1]
            for t in range(len(G_episode)):
                self.buffer[episode][t]['Return'] = G_episode[t]

        A = []
        self.buffer = normalize_buffer_entry(self.buffer, key='rewards')
        for episode in self.buffer: #because we need to calculate those separately!
            V_episode = [step['values'] for step in self.buffer[episode]]
            V_pseudo_terminal = V_episode[-1]
            V_episode.append(V_pseudo_terminal)
            V_episode = np.array(V_episode)

            r_episode = [step['rewards'] for step in self.buffer[episode]]
            r_episode.append(V_pseudo_terminal)
            r_episode = np.array(r_episode)

            deltas = (r_episode[:-1] ) + gamma * V_episode[1:] - V_episode[:-1]
            A_eisode = discount_cumsum(deltas, gamma * gae_lambda)
            A_eisode = A_eisode
            A_eisode = A_eisode.tolist()
            A.extend(A_eisode)
            for t in range(len(A_eisode)):
                self.buffer[episode][t]['advantage'] = A_eisode[t]

        #normalize advantage:
        self.buffer = normalize_buffer_entry(self.buffer, key='advantage')

        # for episode in self.buffer.keys():
        #     for t in range(len.self.buffer[episode]):
        #         self.buffer[episode][t]['advantage'] = self.buffer[episode][t]['advantage'] - entropy_reg * self.buffer[episode][t]['logprob']


        return True

    def fetch_batch(self, batchsize=32, indices=None):

        np.random.shuffle(self.available_indices)
        batch_indices = self.available_indices[:batchsize]

        #ToDo: implement trajectories longer than 1, might be base on same code as DQN buffer

        if not self.multivariate:
            actions_batch = {}
            log_probs_batch = {}
            for action_type in self.action_types:
                actions_batch[action_type] = [self.buffer[sample_episode][transition]['a_taken'][action_type]
                                              for [sample_episode, transition] in batch_indices]

                log_probs_batch[action_type] = [self.buffer[sample_episode][transition]['logprob'][action_type]
                                                for [sample_episode, transition] in batch_indices]
        else:
            actions_batch = [self.buffer[sample_episode][transition]['a_taken']
                                          for [sample_episode, transition] in batch_indices]

            log_probs_batch = [self.buffer[sample_episode][transition]['logprob']
                                            for [sample_episode, transition] in batch_indices]

        states_batch = [self.buffer[sample_episode][transition]['observations']
                        for [sample_episode, transition] in batch_indices]
        advantages_batch = [self.buffer[sample_episode][transition]['advantage']
                        for [sample_episode, transition] in batch_indices]
        returns_batch = [self.buffer[sample_episode][transition]['Return']
                        for [sample_episode, transition] in batch_indices]

        batch = {'actions_taken': actions_batch, #dictionary with a batch f
                 'log_probs': log_probs_batch,
                 'observations': states_batch,
                 'Return': returns_batch,
                 'advantages': advantages_batch
                 }
        return batch