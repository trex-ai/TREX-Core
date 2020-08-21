import tenacity
from _agent._utils.metrics import Metrics

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

    # Core Functions, learn and act, called from outside
    async def learn(self, **kwargs):
        # learn must exist even if unused because participant expects it.
        if not self.learning:
            return

    async def act(self, **kwargs):
        actions = {}

        # empty action dictionary means no trading in the community.
        # defaults to net-metering/net-billing
        if 'storage' not in self.__participant:
            return actions

        next_settle = self.__participant['timing']['next_settle']

        # amount of energy that the agent has to play with.
        generation, load = await self.__participant['read_profile'](next_settle)
        residual_load = load - generation
        residual_gen = -residual_load

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
        return actions

    async def reset(self, **kwargs):
        return True