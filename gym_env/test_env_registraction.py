import gym

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

from gym_env.envs.gym_env import TREXenv
from gym.spaces.utils import flatten_space
from gym.spaces.utils  import flatten, unflatten
env = DummyVecEnv([lambda : TREXenv()])
print(env.envs[0].action_space.sample())