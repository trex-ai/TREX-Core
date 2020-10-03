import tenacity
from _agent._utils.metrics import Metrics
import asyncio
from _agent._rewards import economic_advantage as reward


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
        self._reward = reward.Reward(self.__participant['timing'],
                                     self.__participant['ledger'],
                                     self.__participant['market_info'])

        self.track_metrics = kwargs['track_metrics'] if 'track_metrics' in kwargs else False
        self.metrics = Metrics(self.__participant['id'], track=self.track_metrics)
        if self.track_metrics:
            self.__init_metrics()
        self.bid_price = 0
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
            'participant_id' : pid,
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
        next_settle = self.__participant['timing']['next_settle']
        if 'storage' in self.__participant:
            storage_schedule = self.__participant['storage']['check_schedule'](next_settle)
            # storage_schedule = self.__participant['storage']['schedule'](next_settle)
            max_charge = storage_schedule[next_settle]['energy_potential'][1]
            max_discharge = storage_schedule[next_settle]['energy_potential'][0]

        observations = await self.get_observations()
        generation = observations['observations']['next_settle_gen_value']
        load = observations['observations']['next_settle_load_value']
        residual_load = load - generation
        residual_gen = -residual_load
        rewards = await self._reward.calculate()
        observations['reward'] = rewards
        await self.__participant['emit']('get_remote_actions',
                                         data=observations,
                                         namespace='/simulation')
        await self.wait_for_actions.wait()
        # print(self.next_actions)
        self.bid_price=self.next_actions['actions']['bids']['price']
        actions = {}
        # for action in self.next_actions['actions']:
        #     actions[action] = {str(next_settle): self.next_actions['actions'][action]}

        if residual_load > 0:
            if 'storage' in self.__participant:
                effective_discharge = -min(residual_load, abs(max_discharge))
                actions['bess'] = {str(next_settle): effective_discharge}
            else:
                effective_discharge = 0

            final_residual_load = residual_load + effective_discharge
            if final_residual_load > 0 and self.bid_price:
                actions['bids'] = {
                    str(next_settle): {
                        'quantity': final_residual_load,
                        'source': 'solar',
                        'price': self.bid_price
                    }
                }


            #TODO: need to put the actions['asks']


        if self.track_metrics:
            await asyncio.gather(
                self.metrics.track('timestamp', self.__participant['timing']['current_round'][1]),
                self.metrics.track('actions_dict', actions),
                self.metrics.track('next_settle_load', observations['observations']['next_settle_load_value']),
                self.metrics.track('next_settle_generation', observations['observations']['next_settle_gen_value']))
            # if 'storage' in self.__participant:
            #     await self.metrics.track('storage_soc', projected_soc)

            await self.metrics.save(10000)
        # print(actions)
        return actions

    async def step(self):
        next_actions = await self.act()
        return next_actions

    async def reset(self, **kwargs):
        return True

    def flatten_actions(self, action_dictionary):
        flattened_actions = []

        return flattened_actions

    def __init_metrics(self):
        import sqlalchemy
        '''
        Pretty self explanitory, this method resets the metric lists in 'agent_metrics' as well as zeroing the metrics dictionary. 
        '''
        self.metrics.add('timestamp', sqlalchemy.Integer)
        self.metrics.add('actions_dict', sqlalchemy.JSON)
        self.metrics.add('next_settle_load', sqlalchemy.Integer)
        self.metrics.add('next_settle_generation', sqlalchemy.Integer)
        if 'storage' in self.__participant:
            self.metrics.add('storage_soc', sqlalchemy.Float)
