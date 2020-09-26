import socketio
from gym_env.envs.gym_env import TREXenv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from gym.spaces.utils import unflatten
from baselines.common.models import mlp, get_network_builder
from baselines.ppo2.model import Model
import tensorflow as tf
# from baselines.ppo2.ppo2 import learn
# from baselines.ppo2.runner import Runner


class Controller:
    """
    This is the controller for the gym
    """

    def __init__(self, sio_client, **kwargs):
        self.policy_network = mlp

        self.__client = sio_client
        self.id_list = []
        self.big_gym_energy = DummyVecEnv([lambda : TREXenv()])
        # get the network - TODO: this may be relegated to the config for ease
        if kwargs:
            gym_kwargs = kwargs['gym']
        else:
            gym_kwargs = {}
        policy_fn = get_network_builder('mlp')(**gym_kwargs)
        print(policy_fn)
        self.network = policy_fn(self.big_gym_energy.envs[0].observation_space.shape)
        print('Network initialized', dir(self.network))
        # Setup the model
        self.model = Model(ac_space=self.big_gym_energy.envs[0].action_space,
                           policy_network=self.network,
                           ent_coef=0.0, vf_coef=0.5,
                           max_grad_norm=0.5)


        # action_space = self.big_gym_energy.envs[0].action_space
        # self.model = Model(ac_space=action_space, policy_network=self.policy_network,
        #                    ent_coef=0, vf_coef=0.5, max_grad_norm=0.5)
        # print(self.model, 'Ppo2 model is initialized')
        # self.gamma = 0.99
        # self.lam = 0.95
        # self.runner = Runner(env=self.big_gym_energy, model=self.model, nsteps=1
        #                      , gamma=self.gamma,lam=self.lam)
        # Register client in server

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
        obs = self.big_gym_energy.envs[0].reset()
        print(obs.shape)
        obs = tf.reshape(obs, [1, 200])


        #query the model for the actions
        actions_array, vf, something, neglogp = self.model.train_model.step(obs)
        print('Actions array', actions_array)

        unflattened_dictionary = unflatten(self.big_gym_energy.envs[0]._action_space,actions_array[0].numpy())
        print(unflattened_dictionary)
        unflattened_dictionary['asks']['time_interval']['price'] = unflattened_dictionary['asks']['time_interval']['price'][0]
        unflattened_dictionary['bids']['time_interval']['price'] = unflattened_dictionary['bids']['time_interval']['price'][0]

        # actions needs to be of the form:
        actions = {
                      'participant_id': participant_id,
                      'market_id': market_id,
                  'actions': unflattened_dictionary,

        }
        await self.__client.emit('got_remote_actions', actions, namespace='/simulation')


