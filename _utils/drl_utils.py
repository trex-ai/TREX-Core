import random
import tensorflow as tf
from random import randint
import numpy as np
from _utils import utils
from collections import OrderedDict

# this is a function to robustly pick the argmax in a random fashion if we happen to have several identically maximal values
async def robust_argmax(tensor):
    max_value = tf.reduce_max(tensor)
    max_value_idxs = tf.where(tf.math.equal(max_value, tf.squeeze(tensor, axis=0)))
    random_max_value_idx = tf.random.shuffle(max_value_idxs)[0]
    return random_max_value_idx

#ToDO: change the replay buffer into an ordered dict with the following key as a tuple:
# (pos, episode)
# this will likely be necessary to make sure we do not crossample transition samples that overlap between episodes!
class ExperienceReplayBuffer:
    def __init__(self, max_length=1e4, learn_wait=100):
        self.max_length = max_length
        self.learn_wait = learn_wait
        self.buffer = []
        self.dict_bufffer = OrderedDict()
        # entry = {   'a': None, #actions
        #            's': None, #states
        #            'r': None, #rewards
        #            'episode': None #episode number
        #            }

    def add_entry(self, actions, states, rewards, episode=0, ts=None):
        entry = { 'a': actions, #actions
                   's': states, #states
                   'r': rewards, #rewards
                    'episode': episode
                   }
        self.dict_bufffer[ts] = entry
        self.buffer.append(entry)

        if len(self.buffer) > self.max_length: #make sure its the right length
            self.buffer = self.buffer[-self.max_length:]

    def clear_buffer(self):
        self.buffer = []
        self.dict_bufffer.clear()

    # def match_episodes(self, candidate_index, trajectory_length=1):
    #     start = self.buffer[candidate_index]['episode']
    #     #ToDO: implement trajectory functionality
    #     trajectory = [self.buffer[candidate_index + i + 1]['episode'] for i  in range(trajectory_length)]
    #     same_episode = np.equal(start, trajectory).tolist()

    def fetch_batch_indices(self, batchsize):
        indices = np.random.choice(len(self.buffer) - 1, batchsize, replace=False)
        passed_trajectory_check = 0
        inspected_index_number = 0
        while not passed_trajectory_check < 1.0:  # check to see if we have the followup entry from the same trajectory
            index = indices[inspected_index_number]
            if self.buffer[index]['episode'] == self.buffer[index + 1]['episode']:
                passed_trajectory_check = + 1 / len(indices)
                inspected_index_number = + 1
            else:
                print('.')
                found_replacement = False
                while not found_replacement:
                    replacement_candidate = np.random.choice(len(self.buffer) - 1, 1)
                    if replacement_candidate not in indices:
                        found_replacement = True

        return indices

    def should_we_learn(self):
        if len(self.buffer) > self.learn_wait:
            return True
        else:
            return False

    def fetch_batch(self, batchsize=32, indices=None):

        if indices is None:
            indices =  self.fetch_batch_indices(batchsize)
        batch = {'actions':     [self.buffer[k]['a'] for k in indices],
                 'rewards':     [self.buffer[k]['r'] for k in indices],
                 'states':      np.array([self.buffer[k]['s'] for k in indices]),
                 'next_states': np.array([self.buffer[k+1]['s'] for k in indices])
                 }
        return batch

