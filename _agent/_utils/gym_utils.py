'''
This file contains the helper functions to integrate Trex and the Gym environment

'''

import gym
from gym import spaces

class GymPlug:
    def __init__(self):
        #do stuff


class TREXenv(gym.Env):
    def __init__(self):

    def step(self, actions):
        #this is were we will have t
        return obs, reward, info

    def reset(self):
        # this resets the TREX env -- prolly will have to have this be where the main file is called
        return obs

    def render(self, mode='human', close=False):
        # this renders the environment to the user, for us it will just print shit to the console




