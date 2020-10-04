# from _clients.participants.participants import Residential

import tenacity
from _agent._utils.metrics import Metrics
import asyncio
from _utils import jkson as json
# import serialize


class Trader:
    def __init__(self, **kwargs):
        # Some util stuffies
        self.__participant = kwargs['trader_fns']
        self.status = {
            'weights_loading': False,
            'weights_loaded': False,
            'weights_saving': False,
            'weights_saved': True
        }

        # Initialize the agent learning parameters for the agent (your choice)
        self.bid_price = kwargs['bid_price'] if 'bid_price' in kwargs else None
        self.ask_price =  kwargs['ask_price'] if 'ask_price' in kwargs else None

        # Initialize the metrics, whatever you
        # set learning and track_metrics flags
        self.track_metrics = kwargs['track_metrics'] if 'track_metrics' in kwargs else False
        self.metrics = Metrics(self.__participant['id'], track=self.track_metrics)
        if self.track_metrics:
            self.__init_metrics()

    def __init_metrics(self):
        import sqlalchemy
        '''
        Pretty self explanitory, this method resets the metric lists in 'agent_metrics' as well as zeroing the metrics dictionary. 
        '''
        self.metrics.add('timestamp', sqlalchemy.Integer)
        self.metrics.add('actions_dict', sqlalchemy.JSON)
        self.metrics.add('next_settle_load', sqlalchemy.Integer)
        self.metrics.add('next_settle_generation', sqlalchemy.Integer)

        # if self.battery:
        #     self.metrics.add('battery_action', sqlalchemy.Integer)
        #     self.metrics.add('state_of_charge', sqlalchemy.Float)

    # Core Functions, learn and act, called from outside

    async def act(self, **kwargs):
        # actions are none so far
        # ACTIONS ARE FOR THE NEXT settle!!!!!

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

        actions = {}

        next_settle = self.__participant['timing']['next_settle']
        generation, load = await self.__participant['read_profile'](next_settle)
        residual_load = load - generation
        residual_gen = -residual_load

        if 'storage' in self.__participant:
            storage_schedule = self.__participant['storage']['check_schedule'](next_settle)
            # storage_schedule = self.__participant['storage']['schedule'](next_settle)
            max_charge = storage_schedule[next_settle]['energy_potential'][1]
            max_discharge = storage_schedule[next_settle]['energy_potential'][0]
        # else:
        #     max_charge = 0
        #     max_discharge = 0

        # if were lacking energy, get as much as possible out of battery
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
                # await self.metrics.track('price_action', 0.08)
                # await self.metrics.track('amount_action', final_residual_load)

        # if we have too much, cram as much as possible into battery
        elif residual_gen > 0:
            if 'storage' in self.__participant:
                effective_charge = min(residual_gen, max_charge)
                actions['bess'] = {str(next_settle): effective_charge}
            else:
                effective_charge = 0

            final_residual_gen = residual_gen - effective_charge
            if final_residual_gen > 0 and self.ask_price:
                actions['asks'] = {
                    str(next_settle): {
                        'quantity': final_residual_gen,
                        'source': 'solar',
                        'price': self.ask_price
                    }
                }
                # await self.metrics.track('price_action', 0.12)
                # await self.metrics.track('amount_action', final_residual_gen)
        if self.track_metrics:
            await asyncio.gather(
                self.metrics.track('timestamp', self.__participant['timing']['current_round'][1]),
                self.metrics.track('actions_dict', actions),
                self.metrics.track('next_settle_load', load),
                self.metrics.track('next_settle_generation', generation))

            await self.metrics.save(10000)
        return actions

    async def step(self):
        next_actions = await self.act()
        return next_actions

    async def reset(self, **kwargs):
        return True