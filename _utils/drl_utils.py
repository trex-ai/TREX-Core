import random
import tensorflow as tf
from random import randint
import numpy as np
from TREX_Core._utils import utils
from collections import OrderedDict, Counter
import itertools
import scipy.signal
from tensorflow import keras as k
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow_probability as tfp

def build_hidden_layer(signal, type='FFNN', num_hidden=32, name='Actor', initial_state=None, initializer=k.initializers.HeNormal()):

    if type == 'FFNN':
        signal = k.layers.Dense(num_hidden,
                                         activation="elu",
                                         kernel_initializer=initializer,
                                         name=name)(signal)
        return signal, None
    elif type == 'GRU':
        signal, last_state = k.layers.GRU(num_hidden,
                              activation='tanh',
                                recurrent_activation='sigmoid',
                              kernel_initializer=initializer,
                              return_sequences=True, return_state=True,
                              name=name)(signal, initial_state=initial_state)
        return signal, last_state

    else:
        print('requested layer type (', type, ') not recognized, failed to build ', name)
        return False, False

def build_hidden(internal_signal, inputs, outputs, hidden_actor=[32,32,32], type='FFNN'):
    hidden_layer = 0
    initial_states_dummy = {}
    for num_hidden_neurons in hidden_actor:
        if type == 'GRU':
            initial_state = k.layers.Input(shape=num_hidden_neurons, name='GRU_' + str(hidden_layer) + '_initial_state')
            inputs['GRU_'+str(hidden_layer)+'_state'] = initial_state
            initial_states_dummy['GRU_'+str(hidden_layer)+'_state'] = tf.zeros((1,num_hidden_neurons))

        else:
            initial_state = None
        internal_signal, last_state = build_hidden_layer(internal_signal,
                                       type=type,
                                       num_hidden=num_hidden_neurons,
                                       initial_state=initial_state,
                                       name='Actor_hidden_' + str(hidden_layer))
        if type == 'GRU':
            outputs['GRU_'+str(hidden_layer)+'_state'] = last_state
        hidden_layer += 1

    return internal_signal, inputs, outputs, initial_states_dummy

def build_actor(num_inputs=4, num_actions=3, hidden_actor=[32], actor_type='FFNN'):
    initializer = k.initializers.HeNormal()
    inputs = {}
    outputs = {}

    shape = (num_inputs,) if actor_type != 'GRU' else (None, num_inputs,)
    internal_signal = k.layers.Input(shape=shape, name='Actor_Input')
    inputs['observations'] = internal_signal

    internal_signal,  inputs, outputs, initial_states_dummy = build_hidden(internal_signal, inputs, outputs, hidden_actor, actor_type)

    concentrations = k.layers.Dense(2 * num_actions,
                                    activation='tanh', #ToDo: test tanh vs None
                                    kernel_initializer=initializer,
                                    name='concentrations')(internal_signal)
    concentrations = huber(concentrations)
    outputs['pi'] = concentrations
    actor_model = k.Model(inputs=inputs, outputs=outputs)

    actor_distrib = tfp.distributions.Beta

    out_dict={'model': actor_model,
              'distribution': actor_distrib,
              'initial_states_dummy': initial_states_dummy}

    return out_dict

def build_critic(num_inputs=4, hidden_critic=[32, 32, 32], critic_type='FFNN'):
    initializer = k.initializers.HeNormal()
    inputs = {}
    outputs = {}
    shape = (num_inputs,) if critic_type != 'GRU' else (None, num_inputs,)
    internal_signal = k.layers.Input(shape=shape, name='Critic_Input')
    inputs['observations'] = internal_signal

    internal_signal, inputs, outputs, initial_states_dummy = build_hidden(internal_signal, inputs, outputs, hidden_critic, critic_type)

    value = k.layers.Dense(1,
                           activation='tanh', #ToDo: test tanh vs None
                           kernel_initializer=initializer,
                           name='ValueHead')(internal_signal)
    outputs['value'] = value
    critic_model = k.Model(inputs=inputs, outputs=outputs)

    critic_dict = {'model': critic_model,
                   'initial_states_dummy': initial_states_dummy}
    return critic_dict

def build_actor_critic_models(num_inputs=4,
                              hidden_actor=[32, 32, 32],
                              actor_type='FFNN', #['FFNN', 'GRU'] #ToDo
                              hidden_critic=[32,32,32],
                              critic_type='FFNN', #['FFNN', 'GRU'] #ToDo
                              num_actions=4):
    # needs to return a suitable actor ANN, ctor PDF function and critic ANN
    initializer = tf.keras.initializers.HeNormal()
    actor_dict = build_actor(num_inputs=num_inputs,
                              num_actions=num_actions,
                              hidden_actor=hidden_actor,
                              actor_type=actor_type)
    critic_dict = build_critic(num_inputs=num_inputs,
                              hidden_critic=hidden_critic,
                              critic_type=critic_type)

    return actor_dict, critic_dict

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

def assemble_subdict_batch(list_of_dicts, entries=None): #entries being a list of entries to include
    dict_of_lists = {}
    if entries is not None:
        list_of_dicts = [list_of_dicts[entry] for entry in entries]
    for dict in list_of_dicts:
        for key in dict:
            if key not in dict_of_lists:
                dict_of_lists[key] = [dict[key]]
            else:
                dict_of_lists[key].append(dict[key])
    return dict_of_lists

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

    def add_entry(self, actions_taken, log_probs, values, observations, rewards, critic_states=None, actor_states=None, episode=0):
        entry = {}
        if actions_taken is not None:
            entry['actions_taken'] = actions_taken
        if log_probs is not None:
            entry['log_probs'] = log_probs
        if values is not None:
            entry['values'] = values
        if observations is not None:
            entry['observations'] = observations
        if rewards is not None:
            entry['rewards'] = rewards
        if critic_states is not None:
            entry['critic_states'] = critic_states
        if actor_states is not None:
            entry['actor_states'] = actor_states

        if episode not in self.buffer: #ToDo: we might need to change this for asynch stuff
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

    async def calculate_advantage(self, gamma=0.99, gae_lambda=0.95, normalize=True):

        for episode in self.buffer:
            V_episode = [step['values'] for step in self.buffer[episode]]
            V_pseudo_terminal = V_episode[-1]

            r_episode = [step['rewards'] for step in self.buffer[episode]]
            r_episode.append(V_pseudo_terminal)
            r_episode_array = np.array(r_episode)

            G_episode = discount_cumsum(r_episode_array, gamma)[:-1]
            for t in range(len(G_episode)):
                self.buffer[episode][t]['returns'] = G_episode[t]

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
                self.buffer[episode][t]['advantages'] = A_eisode[t]

        #normalize advantage:
        if normalize:
            self.buffer = normalize_buffer_entry(self.buffer, key='advantages')
        #ToDo: do some research if normalizing rewards here is useful

        return True

    def _fetch_buffer_entry(self, batch_indices, key, subkeys=False, only_first_entry=False):
        #godl: trajectory_start:trajectory_start+self.trajectory_length
        # if subkeys: #for nested buffer entries
        #     fetched_entry = {}
        #     for subkey in subkeys:
        #         fetched_entry[subkey] = [self.buffer[sample_episode][trajectory_start][key][subkey]
        #                                       for [sample_episode, trajectory_start] in batch_indices]
        #
        # else:
        if self.trajectory_length <=1 or only_first_entry:
            fetched_entry = [self.buffer[sample_episode][trajectory_start][key]
                                          for [sample_episode, trajectory_start] in batch_indices]
        else:
            fetched_entry = []
            for [sample_episode, trajectory_start] in batch_indices:
                fetched_trajectory = [self.buffer[sample_episode][trajectory_start + step][key] for step in range(self.trajectory_length)]
                fetched_entry.append(fetched_trajectory)

        return fetched_entry

    def fetch_batch(self, batchsize=32, indices=None,
                    keys=['actions_taken', 'log_probs', 'observations', 'advantages', 'returns'],

                    ):

        np.random.shuffle(self.available_indices)
        batch_indices = self.available_indices[:batchsize]

        #ToDo: implement trajectories longer than 1, might be base on same code as DQN buffer

        batch = {}
        for key in keys:
            batch[key] = self._fetch_buffer_entry(batch_indices,
                                                  key,
                                                  only_first_entry= True if key == 'actor_states' or key == 'critic_states' else False)

        return batch