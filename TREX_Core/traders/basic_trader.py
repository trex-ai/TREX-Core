from TREX_Core import utils


class Trader:
    CHARGE_HOURS_ALLOWED = frozenset(range(8, 17))

    def __init__(self, **kwargs):
        self.participant = kwargs["context"]
        self.bid_price = kwargs.get("bid_price")
        self.ask_price = kwargs.get("ask_price")
        self.action_scenario_history = {}

    async def act(self, **_kwargs):
        actions = {}
        last_settle = self.participant.timing["last_settle"]
        next_settle = self.participant.timing["next_settle"]
        timezone = self.participant.timing["timezone"]
        next_settle_end = utils.timestamp_to_local(next_settle[1], timezone)

        generation, load = await self.participant.read_profile(next_settle)
        residual_load = load - generation
        residual_gen = -residual_load

        storage_limits = await self._get_storage_limits(next_settle)
        if storage_limits is not None:
            actions.update(await self._get_last_settle_bess_action(last_settle))

        actions.update(
            self._build_market_actions(
                next_settle,
                residual_load,
                residual_gen,
                storage_limits,
                next_settle_end.hour in self.CHARGE_HOURS_ALLOWED,
            )
        )

        return actions

    async def _get_storage_limits(self, next_settle):
        if self.participant.storage is None:
            return None

        storage_schedule = await self.participant.storage.check_schedule(next_settle)
        energy_potential = storage_schedule[next_settle]["energy_potential"]
        return {
            "max_charge": energy_potential[1],
            "max_discharge": energy_potential[0],
        }

    async def _get_last_settle_bess_action(self, last_settle):
        scenario_state = self.action_scenario_history.get(last_settle)
        if scenario_state is None:
            return {}

        last_settle_info = await self.participant.ledger.get_settled_info(last_settle)
        actions = self._build_settled_bess_action(
            last_settle, last_settle_info, scenario_state
        )
        stale_round = self.participant.timing["stale_round"]
        self.action_scenario_history.pop(stale_round, None)
        return actions

    def _build_settled_bess_action(self, last_settle, settled_info, scenario_state):
        scenario_handlers = {
            1: self._scenario_one_bess_quantity,
            2: self._scenario_two_bess_quantity,
            3: self._scenario_three_bess_quantity,
            4: self._scenario_four_bess_quantity,
        }
        handler = scenario_handlers.get(scenario_state["scenario"])
        if handler is None:
            return {}
        return {"bess": {str(last_settle): handler(settled_info, scenario_state)}}

    def _scenario_one_bess_quantity(self, settled_info, scenario_state):
        settled_bids = settled_info["bids"]["quantity"]
        return min(
            scenario_state["max_charge"],
            max(0, settled_bids - scenario_state["residual_load"]),
        )

    def _scenario_two_bess_quantity(self, settled_info, scenario_state):
        settled_asks = settled_info["asks"]["quantity"]
        return -min(
            abs(scenario_state["max_discharge"]),
            scenario_state["residual_load"] + settled_asks,
        )

    def _scenario_three_bess_quantity(self, settled_info, scenario_state):
        settled_bids = settled_info["bids"]["quantity"]
        return min(
            scenario_state["max_charge"],
            settled_bids + scenario_state["residual_gen"],
        )

    def _scenario_four_bess_quantity(self, settled_info, scenario_state):
        settled_asks = settled_info["asks"]["quantity"]
        return -min(
            abs(scenario_state["max_discharge"]),
            max(0, settled_asks - scenario_state["residual_gen"]),
        )

    def _build_market_actions(
        self,
        next_settle,
        residual_load,
        residual_gen,
        storage_limits,
        charge_allowed,
    ):
        if residual_load > 0:
            return self._build_load_actions(
                next_settle,
                residual_load,
                residual_gen,
                storage_limits,
                charge_allowed,
            )
        if residual_gen > 0:
            return self._build_generation_actions(
                next_settle,
                residual_load,
                residual_gen,
                storage_limits,
                charge_allowed,
            )
        return {}

    def _build_load_actions(
        self,
        next_settle,
        residual_load,
        residual_gen,
        storage_limits,
        charge_allowed,
    ):
        if storage_limits is None:
            return {
                "bids": {
                    str(next_settle): {
                        "quantity": residual_load,
                        "price": self.bid_price,
                    }
                }
            }

        if charge_allowed:
            self._record_scenario(
                next_settle, 1, residual_load, residual_gen, storage_limits
            )
            return {
                "bids": {
                    str(next_settle): {
                        "quantity": residual_load + storage_limits["max_charge"],
                        "price": self.bid_price,
                    }
                }
            }

        self._record_scenario(
            next_settle, 2, residual_load, residual_gen, storage_limits
        )
        return {
            "asks": {
                "bess": {
                    str(next_settle): {
                        "quantity": max(
                            0,
                            storage_limits["max_discharge"] - residual_load,
                        ),
                        "price": self.ask_price,
                    }
                }
            }
        }

    def _build_generation_actions(
        self,
        next_settle,
        residual_load,
        residual_gen,
        storage_limits,
        charge_allowed,
    ):
        if storage_limits is None:
            return {
                "asks": {
                    "solar": {
                        str(next_settle): {
                            "quantity": residual_gen,
                            "price": self.ask_price,
                        }
                    }
                }
            }

        if charge_allowed:
            self._record_scenario(
                next_settle, 3, residual_load, residual_gen, storage_limits
            )
            return {
                "bids": {
                    str(next_settle): {
                        "quantity": max(0, storage_limits["max_charge"] - residual_gen),
                        "price": self.bid_price,
                    }
                }
            }

        self._record_scenario(
            next_settle, 4, residual_load, residual_gen, storage_limits
        )
        return {
            "asks": {
                "solar": {
                    str(next_settle): {
                        "quantity": residual_gen,
                        "price": self.ask_price,
                    }
                },
                "bess": {
                    str(next_settle): {
                        "quantity": storage_limits["max_discharge"],
                        "price": self.ask_price,
                    }
                },
            }
        }

    def _record_scenario(
        self,
        next_settle,
        scenario,
        residual_load,
        residual_gen,
        storage_limits,
    ):
        self.action_scenario_history[next_settle] = {
            "scenario": scenario,
            "residual_load": residual_load,
            "residual_gen": residual_gen,
            **storage_limits,
        }

    async def step(self):
        return await self.act()

    async def reset(self, **_kwargs):
        self.action_scenario_history.clear()
        return True
