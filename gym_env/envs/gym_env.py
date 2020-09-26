import gym
from gym import spaces
from gym.spaces.utils import flatten, unflatten, flatten_space

class TREXenv(gym.Env):
    def __init__(self):
        super(TREXenv,self).__init__()
        # this action space needs to be defined in the same way that we pass actions to the simulation
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

        self._action_space = spaces.Dict({
            'bids':spaces.Dict({
                'time_interval':spaces.Dict({
                    'quantity':spaces.Discrete(100),
                    'source':spaces.Discrete(3),
                    'price':spaces.Box(
                        low=0.0,
                        high=10,
                        shape=(1,),
                        dtype=float
                    )
                })
            }),
            'asks':spaces.Dict({
                'time_interval': spaces.Dict({
                    'quantity': spaces.Discrete(100),
                    'source': spaces.Discrete(3),
                    'price': spaces.Box(
                        low=0.0,
                        high=10,
                        shape=(1,),
                        dtype=float
                    )
                })
            })
        })
        self.action_space = flatten_space(self._action_space)


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
            'next_settle_load_value': spaces.Discrete(100),
            'next_settle_gen_value': spaces.Discrete(100)
        })
        self.observation_space = flatten_space(self._observation_space)


    def step(self, actions):
        #this is were we will have the
        obs = self.observations
        reward = 0
        dones = False
        info = {}
        print("actions in gym", actions)
        return obs, reward, dones, info

    def reset(self):
        # this resets the TREX env -- prolly will have to have this be where the main file is called
        #
        _obs = self._observation_space.sample()
        print(_obs)
        obs = flatten(self._observation_space, _obs)
        print(obs)
        return obs

    def render(self, mode='human', close=False):
        return True
        # this renders the environment to the user, for us it will just print shit to the console

    def get_observations(self,observations):
        # this is where we set the values that need to be
        self.observations = observations

