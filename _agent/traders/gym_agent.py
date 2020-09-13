'''
This is the gym plug api for TREX. It needs to have the following 3 methods:
1. __init__
2. act
    this one needs to return an action
3. learn
    this is simply a holder since
'''
import asyncio
from _utils.gym_utils import GymPlug
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.ppo2.ppo2 import learn

# env3 = DummyVecEnv([lambda : gym.make('TestEnv-v0')])
class Trader:
    def __init__(self):
        '''
        There will need to be some initialization of

        You should initialize both a baselines model and the gym env.

        Gym agent will have to tell the gym_client if is
        '''
        self.Genv = DummyVecEnv([lambda: gym.make('GymEnv-v0')])

        self.learnfunction = learn # so this is just a placeholder so that in act it can run the function

    #REMOTE AGENT STUFF STOLEN FROM DQN agent
    # def update_remote_agent(self, client):
    #     '''
    #     Updates the remote agent with the client that was passed to it and calls asyncio.Event().set()
    #     '''
    #     self.__remote_agent = {
    #         'client': client,
    #         'policy': {
    #             'get': asyncio.Event(),
    #             'buffer': {}
    #         }
    #     }
    #     self.__remote_agent['policy']['get'].set()
    #     # print('remote agent set', self.__remote_agent)
    async def act(self):
        # actions = {
        #     'bess': {
        #         time_interval: scheduled_qty
        #     },
        #     'bids': {
        #         time_interval: {
        #             'quantity': qty,
        #             'source': source,
        #             'price': dollar_per_kWh
        #         }
        #     },
        #     'asks': {
        #         time_interval: {
        #             'quantity': qty,
        #             'source': source,
        #             'price': dollar_per_kWh?
        #         }
        #     }
        # }

        #TODO: pass a message
        actions = {}

        # Bid or ask
        actions['bids'] = {}
        return actions

    async def learn(self):
        # this is where you pass out the collected
        return None