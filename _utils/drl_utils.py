import random
import tensorflow as tf
from random import randint
import numpy as np
from _utils import utils
from collections import OrderedDict, Counter

# this is a function to robustly pick the argmax in a random fashion if we happen to have several identically maximal values
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

class ExperienceReplayBuffer_New:
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
        #FixMe: this does not work in parallel acess cases!!!


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

