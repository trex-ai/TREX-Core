import socketio
from gym_env.envs.gym_env import TREXenv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

class Controller:
    """
    This is the controller for the gym
    """

    def __init__(self, sio_client, **kwargs):

        self.__client = sio_client
        self.id_list = []
        self.big_gym_energy = DummyVecEnv([lambda : TREXenv()])
        print("gym controller initialized")

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

        #this needs to be
        participant_id = message.pop('participant_id')
        market_id = message.pop('market_id')
        observations = message.pop('observations')
        print(participant_id)

        # print(self.big_gym_energy.envs[0].action_space.sample())

        #keep track of ids:
        # This TREX < -gym < -baselines

        action_dict = self.big_gym_energy.envs[0]._action_space.sample()  # this should be an action dictionary, along with the agent that the actions should be routed to.
        #

        # actions needs to be of the form:
        actions = {
                      'participant_id': participant_id,
                      'market_id': market_id,
                  'actions': action_dict,

        }
        await self.__client.emit('got_remote_actions',actions, namespace='/simulation')


