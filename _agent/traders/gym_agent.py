# from _clients.participants.participants import Residential

import tenacity
from TREX_Core._agent._utils.metrics import Metrics
import asyncio
from TREX_Core._utils import jkson as json
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

        # TODO: Find out where the action space will be defined: I suspect its not here
        # Initialize the agent learning parameters for the agent (your choice)
        # self.bid_price = kwargs['bid_price'] if 'bid_price' in kwargs else None
        # self.ask_price = kwargs['ask_price'] if 'ask_price' in kwargs else None

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
        #             'price': dollar_per_kWh
        #         }
        #     },
        #     'asks': {
        #         source:{
        #              time_interval: {
        #                 'quantity': qty,
        #                 'price': dollar_per_kWh?
        #              }
        #          }
        #     }
        # }

        actions = {}
        # Pre transition information.
        next_settle = self.__participant['timing']['next_settle']
        generation, load = await self.__participant['read_profile'](next_settle)
        residual_load = load - generation
        residual_gen = -residual_load
        message_data = {
            'next_settle' : next_settle,
            'generation': generation,
            'load' : load,
            'residual_load': residual_load,
            'residual_get': residual_gen
        }
        # TODO: send out pre transition information to the envController
        await self.__participant['emit']('pre_transition_data', message_data, namespace='/simulation')

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

    async def decode_actions(self, action_indices: dict, next_settle):
        actions = dict()
        # print(action_indices)

        price = self.actions['price'][action_indices['price']]
        quantity = self.actions['quantity'][action_indices['quantity']]

        if quantity > 0:
            actions['bids'] = {
                str(next_settle): {
                    'quantity': quantity,
                    'price': price
                }
            }
        elif quantity < 0:
            actions['asks'] = {
                'solar': {
                    str(next_settle): {
                        'quantity': -quantity,
                        'price': price
                    }
                }
            }

        if 'storage' in self.actions:
            target = self.actions['storage'][action_indices['storage']]
            if target:
                actions['bess'] = {
                    str(next_settle): target
                }
        # print(actions)

        #log actions for later histogram plot
        for action in self.actions:
            self.episode_actions[action].append(self.actions[action][action_indices[action]])
        return actions