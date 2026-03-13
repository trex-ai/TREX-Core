from TREX_Core import utils


class Trader:
    """Baseline trader for net-metering/net-billing self-sufficiency."""

    def __init__(self, **kwargs):
        self.__participant = kwargs["trader_fns"]
        self.track_metrics = kwargs.get("track_metrics", False)

    async def act(self, **_kwargs: object) -> dict[str, dict[str, float]]:
        actions: dict[str, dict[str, float]] = {}

        # empty action dictionary means no trading in the community.
        # defaults to net-metering/net-billing
        if "storage" not in self.__participant:
            return actions

        next_settle = self.__participant["timing"]["next_settle"]
        timezone = self.__participant["timing"]["timezone"]
        next_settle_end = utils.timestamp_to_local(next_settle[1], timezone)

        # amount of energy that the agent has to play with.
        generation, load = await self.__participant["read_profile"](next_settle)
        residual_load = load - generation
        residual_gen = -residual_load

        # if battery exists, then
        # get the battery information:
        storage_schedule = await self.__participant["storage"]["check_schedule"](
            next_settle
        )
        max_charge = storage_schedule[next_settle]["energy_potential"][1]
        max_discharge = storage_schedule[next_settle]["energy_potential"][0]

        # if were lacking energy, get as much as possible out of battery

        if residual_load > 0:
            effective_discharge = -min(residual_load, abs(max_discharge))
            if next_settle_end.hour not in (8, 9, 10, 11, 12, 13, 14, 15, 16):
                # only allow discharge from 5PM to 7AM
                actions["bess"] = {str(next_settle): effective_discharge}

        # if we have too much generation, charge the battery as much as possible
        elif residual_gen > 0:
            effective_charge = min(residual_gen, max_charge)
            actions["bess"] = {str(next_settle): effective_charge}
        return actions

    async def step(self) -> dict[str, dict[str, float]]:
        return await self.act()
