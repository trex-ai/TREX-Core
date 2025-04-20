import asyncio
# import tenacity
# from TREX_Core.utils.records import Metrics


class Trader:
    """The baseline trader that emulates behaviour under net-metering/net-billing with a focus on self-sufficiency
    """
    def __init__(self, **kwargs):
        self.participant = kwargs['context']
        # -------- hook registry -----------
        self._act_hooks = list()

        # one‑time capability discovery
        if self.participant.storage is not None:
            self._act_hooks.append(self._act_storage_hook)

    async def _act_storage_hook(self, next_settle, residual_gen, residual_load):
        storage_schedule = await self.participant.storage.check_schedule(next_settle)
        max_charge = storage_schedule[next_settle]['energy_potential'][1]
        max_discharge = storage_schedule[next_settle]['energy_potential'][0]

        # if were lacking energy, get as much as possible out of battery
        if residual_load > 0:
            effective_discharge = -min(residual_load, abs(max_discharge))
            return {'bess': {str(next_settle): effective_discharge}}
        # if we have too much generation, charge the battery as much as possible
        elif residual_gen > 0:
            effective_charge = min(residual_gen, max_charge)
            return {'bess': {str(next_settle): effective_charge}}
        return {}

    async def act(self, **kwargs):
        actions = dict()
        next_settle = self.participant.timing['next_settle']
        generation, load = await self.participant.read_profile(next_settle)
        residual_load = load - generation
        residual_gen = -residual_load

        for hook in self._act_hooks:
            add = await hook(next_settle, residual_gen, residual_load)
            actions.update(add)
        return actions

    async def step(self):
        next_actions = await self.act()
        return next_actions
