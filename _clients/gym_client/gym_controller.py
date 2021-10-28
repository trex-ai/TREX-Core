import socketio
import asyncio
import concurrent.futures
from gym_env.envs.gym_env import TREXenv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from gym.spaces.utils import unflatten, flatten
from baselines.common.models import mlp, get_network_builder
from baselines.ppo2.model import Model
import tensorflow as tf
import numpy as np
from baselines.ppo2.ppo2 import learn
# from baselines.ppo2.runner import Runner
import functools
import gym


class Controller:
    """
    This is the controller for the gym
    """

    def __init__(self, sio_client, **kwargs):
        self.policy_network = mlp
        self._market = None
        self.is_learning = False
        self.__client = sio_client
        self.id_list = []
        self.big_gym_energy = DummyVecEnv([lambda : TREXenv()])

        self.MCC= DummyVecEnv([lambda : gym.make('MountainCarContinuous-v0')])
        print(self.MCC)
        self.counter = 0
        # get the network - TODO: this may be relegated to the config for ease
        if kwargs:
            gym_kwargs = kwargs['gym']
        else:
            gym_kwargs = {}
        policy_fn = get_network_builder('mlp')(**gym_kwargs)

        self.network = policy_fn(self.big_gym_energy.envs[0].observation_space.shape)
        self.networkMCC = policy_fn(self.MCC.envs[0].observation_space.shape)

        # Setup the model
        self.model = Model(ac_space=self.big_gym_energy.envs[0].action_space,
                           policy_network=self.network,
                           ent_coef=0.0, vf_coef=0.5,
                           max_grad_norm=0.5)

        self.modelMMC = Model(ac_space=self.MCC.envs[0].action_space,
                              policy_network=self.networkMCC,
                              ent_coef=0.0, vf_coef=0.5,
                              max_grad_norm=0.5)



    async def register(self):
        client_data = {
            'type': ('remote_agent', '')
        }
        print('We registered the gym _client')
        await self.__client.emit('register', client_data, namespace='/simulation')

    async def get_remote_actions(self, message):
        # this  TREX->gym->baselines
        # message will consist of this:

        participant_id = message.pop('participant_id')
        market_id = message.pop('market_id')
        observations = message.pop('observations')

        reward = message.pop('reward')
        # print('reward from message', reward)
        # obs = self.big_gym_energy.envs[0].reset()

        # need to flatten obs,
        # TODO: I wonder if this flattening is a problem -- Does baselines flatten??
        flat_observations = flatten(self.big_gym_energy.envs[0]._observation_space, observations)

        if reward is not None:
            # FIXME: this breaks when the value of the reward is 0.0
            self.big_gym_energy.envs[0].send_to_gym(flat_observations,reward)

        obs = tf.reshape(flat_observations, [1, len(flat_observations)])

        # query the model for the actions
        # the model does not like the unflattened observations;
        # TODO: I wonder if I should send these to the agent for metrics
        print(self.model.train_model)
        actions_array, vf, something, neglogp = self.model.train_model.step(obs)

        actions_array = actions_array.numpy()


        actions_dictionary = {
            'bids': {'price' : float(actions_array[0][0])}
        }


        # vaalss = [[0.0,100.0], [0.0,2.0], [0.0, 2.0], [0.0,100.0],[0.0,2.0],[0.0, 2.0]]
        # actions_array = await self.check_actions(actions_array, vaalss)

        # actions_dictionary = {
        #     'bids': {'quantity': int(round(actions_array[0][0])),
        #              'source': self.unmap_source(int(round(actions_array[0][1]))),
        #              'price': float(actions_array[0][2])},
        #     'asks': {'quantity': int(round(actions_array[0][3])),
        #              'source': self.unmap_source(int(round(actions_array[0][4]))),
        #              'price': float(actions_array[0][5])}
        # }

        # actions needs to be of the form:
        actions = {
                      'participant_id': participant_id,
                      'market_id': market_id,
                  'actions': actions_dictionary,

        }
        await self.__client.emit('got_remote_actions', actions, namespace='/simulation')

    async def check_actions(self, action_array, aceptable_val):

        if self.big_gym_energy.envs[0].action_space.contains(action_array):
            return action_array

        for i in range(len(action_array[0])):
            val = action_array[0][i]

            if val < aceptable_val[i][0]:
                #less than
                action_array[0][i] = aceptable_val[i][0]
            if val > aceptable_val[i][1]:
                #greater than
                action_array[0][i] = aceptable_val[i][1]
        return action_array

    def unmap_source(self, source):

        if source == 0:
            return 'grid'
        if source == 1:
            return 'solar'
        if source == 2:
            return 'bess'


    async def emit_go(self,message):
        if self.is_learning:
            return False
        print('emit_go', message)
        self._market = message

        await self.__client.emit('remote_agent_ready',message ,namespace='/simulation')


    async def learn(self):
        import os
        self.is_learning = True
        print('gym_controller learn but not  baselines learn', self.big_gym_energy.envs[0].counter)
        model_path = os.getcwd() + '\_simulations\_models '
        loop = asyncio.get_running_loop()
        self.big_gym_energy.envs[0].normalize_observations()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            self.model = await loop.run_in_executor(pool, functools.partial(learn, network=self.networkMCC, env=self.MCC,
                                                               total_timesteps=4000,
                                                                nsteps=2048, seed=42))

        self.big_gym_energy.envs[0].reset()
        self.is_learning = False
