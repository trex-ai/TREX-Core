'''
This file contains the helper functions to integrate Trex and the Gym environment

Make this the remote client server so it is its own client
'''

import gym
from gym import spaces
import sys

class GymPlug:
    """
    I have a feeling that this gym plug should actually become the gym_runner and interface directly
    with the baselines code
    """
    def __init__(self, agents ):








class TREXenv(gym.Env):
    def __init__(self):
        # this action space needs to be defined in the same way that we pass actions to the simulation
        # Example
        # usage[nested]:
        # self.nested_observation_space = spaces.Dict({
        #     'sensors': spaces.Dict({
        #         'position': spaces.Box(low=-100, high=100, shape=(3,)),
        #         'velocity': spaces.Box(low=-1, high=1, shape=(3,)),
        #         'front_cam': spaces.Tuple((
        #             spaces.Box(low=0, high=1, shape=(10, 10, 3)),
        #             spaces.Box(low=0, high=1, shape=(10, 10, 3))
        #         )),
        #         'rear_cam': spaces.Box(low=0, high=1, shape=(10, 10, 3)),
        #     }),
        #     'ext_controller': spaces.MultiDiscrete((5, 2, 2)),
        #     'inner_state': spaces.Dict({
        #         'charge': spaces.Discrete(100),
        #         'system_checks': spaces.MultiBinary(10),
        #         'job_status': spaces.Dict({
        #             'task': spaces.Discrete(5),
        #             'progress': spaces.Box(low=0, high=100, shape=()),
        #         })
        #     })
        # })
        # TREX action space:
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
        # FIXME: sept21/2020 dictionaries dont work.
        #actions need to now be not in dictionaries, arrays will probably be the way [[ask], [bid], [bess]]
        # the other way that I can do this is to fall back on the method we had for DQN, where 
        self.action_space = spaces.Dict({
            "bids" : spaces.Dict({
                spaces.Dict({
                    'time_interval' : spaces.Dict({
                      'quantity' : spaces.Discrete(sys.maxsize), #this is a integer value of watt hours.
                      'source' : spaces.Discrete(3) # This is a numerical representation of the sources: 0 grid, 1 solar, 2 bess
                        ,
                      'price' : spaces.Box(
                            low=0.0,
                            high=10.0,
                            shape=(1,),
                            dtype=float,
                        )
                    })

            })
            }),
            'asks' :spaces.Dict({
                spaces.Dict({
                    'time_interval': spaces.Dict({
                        'quantity': spaces.Box(
                            low=,
                            high=,
                            shape=(1,),
                            dtype=int  #This is a int, so i may want to replace it with a DIsc
                        ),
                        'source': spaces.Box(
                            # FIXME: Figure out if source is a string or a number -> GYM does not seem to have a string space, so that may be a problem
                            low=,
                            high=,
                            shape=(1,),
                            dtype=float
                        ),
                        ''
                        'price': spaces.Box(
                            low=0.0,
                            high=10.0,
                            shape=(1,),
                            dtype=float,
                        )
                    })
            }),
            }),
            #TODO: steven says that battery makes things 1000000000000x harder, so ignore it for now
            # 'bess' : spaces.Dict({
            #     spaces.Dict({
            #         'time_interval' : spaces.Box(
            #             shape=(1,),
            #             dtype=float # FIXME: look into the bess function check_schedule for advice on how to get min and max values
            #         )
            #     })
            #
            # }),

        })

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

        self.observation_space = spaces.Dict({

        })

    def step(self, actions):
        #this is were we will have the

        return obs, reward, dones, info

    def reset(self):
        # this resets the TREX env -- prolly will have to have this be where the main file is called
        #
        return obs

    def render(self, mode='human', close=False):
        # this renders the environment to the user, for us it will just print shit to the console





