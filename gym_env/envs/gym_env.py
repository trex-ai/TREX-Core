import gym
from gym import spaces
from gym.spaces.utils import flatten, flatten_space
# import numpy as np

class TREXenv(gym.Env):
    def __init__(self):
        super(TREXenv,self).__init__()
        # this action space needs to be defined in the same way that we pass actions to the simulation
        # TREX action space:
        self.seed(seed=42)
        self.observations = []
        self.counter = 0
        self.dones = []

        #Action space for 3 params
        # self.action_space = spaces.Box(low=np.array([0.0, 0.0, 0,0.0, 0.0, 0]),
        #                                high=np.array([100.0, 2.0, 2.0, 100.0, 2.0, 2.0]))

        # action space for just price
        self.action_space=spaces.Box(low=0.0, high=2.0, shape=(1,))
        # this should probably also be some dictionary;
        # based on DQN, these are the observations that we used for it:
        # float: time SIN,
        # float: time COS,
        #
        # float: next settle gen value,
        # float: moving average 5 min next settle gen,
        # float: moving average 30 min next settle gen,
        # float: moving average 60 min next settle gen,
        #
        # float: next settle load value,
        # float: moving average 5 min next settle load,
        # float: moving average 30 min next settle load,
        # float: moving average 60 min next settle load,
        #
        # float: next settle projected SOC,
        # float: Scaled battery max charge,
        # float: scaled battery max discharge]

        self._observation_space = spaces.Dict({
            'next_settle_load_value': spaces.Discrete(200),
            'next_settle_gen_value': spaces.Discrete(200)
        })
        self.observation_space = flatten_space(self._observation_space)
        self.index = 0
        self.reward = []


    def step(self, actions):

        # this is were we will have the runner cycle through the experience trajectory.
        # the step function should simply increment an index all the way to the end of the generation
        # print(len(self.observations))
        # print('index', self.index)
        # print(self.observations)
        obs = self.observations[self.index]
        reward = self.reward[self.index]
        dones = 0
        info = {}
        # print("actions in gym", actions)
        self.index += 1
        return obs, reward, dones, info

    def reset(self):
        # this resets the TREX env -- prolly will have to have this be where the main file is called
        #
        self.index = 0
        self.counter = 0
        # self.reward = []
        # self.observations = []

        _obs = self._observation_space.sample()
        obs = flatten(self._observation_space, _obs)

        return obs

    def render(self, mode='human', close=False):
        return True
        # this renders the environment to the user, for us it will just print shit to the console

    def get_observations(self,observations):
        # this is where we set the values that need to be
        self.observations.append(observations)

    def send_to_gym(self, observations, reward):
        self.observations.append(observations)
        self.reward.append(reward)
        self.counter += 1

