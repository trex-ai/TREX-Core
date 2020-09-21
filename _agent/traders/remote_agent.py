import tenacity
from _agent._utils.metrics import Metrics
import asyncio

class Trader:
    """The baseline trader that emulates behaviour under net-metering/net-billing with a focus on self-sufficiency
    """
    def __init__(self, **kwargs):
        self.__participant = kwargs['trader_fns']
        self.status = {
            'weights_loading': False,
            'weights_loaded': False,
            'weights_saving': False,
            'weights_saved': True
        }

        # Initialize the agent learning parameters for the agent (your choice)
        self.agent_data = {}
        self.learning = False
        self.track_metrics = kwargs['track_metrics'] if 'track_metrics' in kwargs else False

        self.next_actions = {}
        self.wait_for_actions = asyncio.Event()
    # Core Functions, learn and act, called from outside
    # async def learn(self, **kwargs):
    #     # learn must exist even if unused because participant expects it.
    #     if not self.learning:
    #         return

    async def get_observations(self):

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

        next_settle = self.__participant['timing']['next_settle']
        generation, load = await self.__participant['read_profile'](next_settle)
        message = {
            'id' : pid,
            'market_id': mid,
            'observations': {
                #observations stuff
                'next_settle_load_value': load,
                'next_settle_gen_value':generation

            }
        }
        return message

    async def act(self, **kwargs):
        """
        query remote agent client for next actions
        Args:
            **kwargs:

        Returns:

        """

        self.next_actions.clear()
        self.wait_for_actions.clear()
        print('get_action', self.next_actions)

        observations = await self.get_observations()
        await self.__participant['emit']('get_remote_actions',
                                         data=observations,
                                         namespace='/simulation')
        await self.wait_for_actions.wait()

        print('got_action', self.next_actions)
        print('-----')
        return self.next_actions

    async def step(self):
        next_actions = await self.act()
        return next_actions

    async def reset(self, **kwargs):
        return True