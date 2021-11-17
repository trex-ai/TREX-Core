'''
This is the gym plug api for TREX. It needs to have the following 3 methods:
1. __init__
2. act
    this one needs to return an action
3. learn
    this is simply a holder since
'''
import asyncio
# from _utils._agent.gym_utils import GymPlug
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.ppo2.ppo2 import learn
import numpy as np
import datetime

# env3 = DummyVecEnv([lambda : gym.make('TestEnv-v0')])
class Trader:
    def __init__(self, **kwargs):
        '''
        There will need to be some initialization of

        You should initialize both a baselines model and the gym env.


        '''
        self.__participant = kwargs['trader_fns']

        self.status = {
            'weights_loading': False,
            'weights_loaded': False,
            'weights_saving': False,
            'weights_saved': True
        }
        self.next_actions = {}
        self.wait_for_actions = asyncio.Event()
        self.learning = False
        self.track_metrics = kwargs['track_metrics'] if 'track_metrics' in kwargs else False



    async def _act(self):
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

        # cleaning
        self.next_actions.clear()
        self.wait_for_actions.clear()

        # get the observations:
        observations = await self._get_observations()

        # send the message to the gym controller asking for actions
        print('Getting actions', self.next_actions)
        await self.__participant['emit']('get_remote_actions',
                                         data=observations,
                                         namespace='/simulation')

        await self.wait_for_actions.wait()
        print("got actions", self.next_actions)
        print("------breakline------")

        return self.next_actions

    async def _get_observations(self):

        pid = self.__participant['id']
        mid = self.__participant['market_id']
        # observations needs id and the observations
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
        last_settle = self.__participant['timing']['last_settle']
        hour = datetime.datetime.utcfromtimetimestamp(last_settle[0]).hour
        min = datetime.datetime.utcfromtimetimestamp(last_settle[0]).minute
        daytime = hour * 60 + min
        max_daytime = 24 * 60
        daytime_in_rad = 2 * np.pi * (daytime / max_daytime)
        day_sin = np.sin(daytime_in_rad)
        day_cos = np.cos(daytime_in_rad)
        print(day_sin)
        print(day_cos)
        next_settle = self.__participant['timing']['next_settle']
        generation, load = await self.__participant['read_profile'](next_settle)
        message = {
            'id' : pid,
            'market_id': mid,
            'observations': {
                # observations stuff
                'next_settle_load_value': load,
                'next_settle_gen_value': generation,
                'last_settle_daytime_sin': day_sin,
                'last_settle_daytime_cos': day_cos

            }
        }
        return message

    async def _learn(self):
        # TODO:this is where we may have to do some sillyness later if we make the gymplug the gymrunner
        return True

    async def step(self):
        """
        This method calls the act and learn processes. Step wraps them so that it is more in line with the
        syntax of gym.
        Returns:

        """
        next_actions = await self.get_actions()

        return next_actions

    async def reset(self):
        return True

    async def get_actions(self):
        '''
        This method asks the envcontroller to pass the right agents action for this step. 
        The env controller should return the values corresponding to the action for the agent to pass to the trader.step() method
        This will have to be called in the step function 
        '''
        #TODO: send a message to the env_controller requesting this agents actions. 
        #The esiest way to get this to work is to bind every agent to their ID in the namespace and use those in a dictionary
        # in the env_controller 
        return action