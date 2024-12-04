import asyncio
# import tenacity
from TREX_Core._agent._utils.metrics import Metrics


class Trader:
    """The baseline trader that emulates behaviour under net-metering/net-billing with a focus on self-sufficiency
    """
    def __init__(self, **kwargs):
        self.__participant = kwargs['trader_fns']
        self.track_metrics = kwargs['track_metrics'] if 'track_metrics' in kwargs else False

        if self.track_metrics:
            self.metrics = Metrics(self.__participant['id'], track=self.track_metrics)
            self.__init_metrics()

    def __init_metrics(self):
        import sqlalchemy
        '''
        Initializes metrics to record into database
        '''
        self.metrics.add('timestamp', sqlalchemy.Integer)
        self.metrics.add('actions_dict', sqlalchemy.JSON)
        # self.metrics.add('rewards', sqlalchemy.Float)
        self.metrics.add('next_settle_load', sqlalchemy.Integer)
        self.metrics.add('next_settle_generation', sqlalchemy.Integer)
        if 'storage' in self.__participant:
            self.metrics.add('storage_soc', sqlalchemy.Float)

    async def act(self, **kwargs):
        actions = {}

        # empty action dictionary means no trading in the community.
        # defaults to net-metering/net-billing
        # if 'storage' not in self.__participant:
        #     return actions

        next_settle = self.__participant['timing']['next_settle']

        # amount of energy that the agent has to play with.
        generation, load = await self.__participant['read_profile'](next_settle)
        residual_load = load - generation
        residual_gen = -residual_load
        if 'storage' in self.__participant:
            # if battery exists, then
            # get the battery information:
            storage_schedule = await self.__participant['storage']['check_schedule'](next_settle)
            max_charge = storage_schedule[next_settle]['energy_potential'][1]
            max_discharge = storage_schedule[next_settle]['energy_potential'][0]

            # if were lacking energy, get as much as possible out of battery
            if residual_load > 0:
                effective_discharge = -min(residual_load, abs(max_discharge))
                actions['bess'] = {str(next_settle): effective_discharge}

            # if we have too much generation, charge the battery as much as possible
            elif residual_gen > 0:
                effective_charge = min(residual_gen, max_charge)
                actions['bess'] = {str(next_settle): effective_charge}

        if self.track_metrics:
            await asyncio.gather(
                self.metrics.track('timestamp', self.__participant['timing']['current_round'][1]),
                self.metrics.track('actions_dict', actions),
                self.metrics.track('next_settle_load', load),
                self.metrics.track('next_settle_generation', generation))
            if 'bess' in actions:
                await self.metrics.track('storage_soc', self.__participant['storage']['info']()['state_of_charge'])
        return actions

    async def step(self):
        next_actions = await self.act()
        return next_actions
