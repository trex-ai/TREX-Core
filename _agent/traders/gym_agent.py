# from _clients.participants.participants import Residential

import tenacity
from TREX_Core._agent._utils.metrics import Metrics
import asyncio
from TREX_Core._utils import jkson as json
# import serialize
from multiprocessing import shared_memory


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
        # Set up the shared lists for data transfer
        print(self.__participant)
        self.name = self.__participant['id']
        action_list_name = self.name + "_actions"
        observation_list_name = self.name + "_obs"

        '''
        Shared lists get initialized on TREXENV side, so all that the agents have to do is connect to their respective 
        observation and action lists. Agents dont have to worry about making the actions pretty, they just have to send
        them into the buffer. 
        '''
        self.shared_list_action = shared_memory.ShareableList(name= action_list_name)
        self.shared_list_observation = shared_memory.ShareableList(name = observation_list_name)


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
        """


        """

        '''
        actions are none so far
        ACTIONS ARE FOR THE NEXT settle!!!!!

        actions = {
            'bess': {
                time_interval: scheduled_qty
            },
            'bids': {
                time_interval: {
                    'quantity': qty,
                    'price': dollar_per_kWh
                }
            },
            'asks': {
                source:{
                     time_interval: {
                        'quantity': qty,
                        'price': dollar_per_kWh?
                     }
                 }
            }
        sources inclued: 'solar', 'bess'
        Actions in the shared list 
        [bid price, bid quantity, solar ask price, solar ask quantity, bess ask price, bess ask quantity]
        }
        '''

        actions = {}
        bid_price = 0.0
        bid_quantity = 0.0
        solar_ask_price = 0.0
        solar_ask_quantity = 0.0
        bess_ask_price = 0.0
        bees_ask_quantity = 0.0

        # Observation information
        next_settle = self.__participant['timing']['next_settle']
        generation, load = await self.__participant['read_profile'](next_settle)
        residual_load = load - generation
        residual_gen = -residual_load
        # Artifact from single agent
        # message_data = {
        #     'next_settle' : next_settle,
        #     'generation': generation,
        #     'load' : load,
        #     'residual_load': residual_load,
        #     'residual_get': residual_gen
        # }
        # TODO: write the pre_transition data to obs buffer
        obs = [next_settle[1], generation, load, residual_load, residual_gen]
        print('Observations', obs)
        await self.write_observation_values(obs)

        # wait for the actions to come from EPYMARL
        await self.read_action_values()
        # actions come in with a set order, they will need to be split up

        # TODO: these need to be set and coded
        # Bid related asks
        bid_price = self.actions[0]
        bid_quantity = self.actions[1]

        # Solar related asks
        solar_ask_price = self.actions[2]
        solar_ask_quantity = self.actions[3]
        #Bess related asks
        bess_ask_price = self.actions[4]
        bees_ask_quantity = self.actions[5]

        if bid_price and bid_quantity:

        # if generation:
        #
        #     actions ={
        #         "asks":{next_settle:{
        #             'quantity':quantity,
        #             'price': user_actions
        #         }
        #         }
        #     }

        if self.track_metrics:
            await asyncio.gather(
                self.metrics.track('timestamp', self.__participant['timing']['current_round'][1]),
                self.metrics.track('actions_dict', actions),
                self.metrics.track('next_settle_load', load),
                self.metrics.track('next_settle_generation', generation))

            await self.metrics.save(10000)
        return actions


    async def step(self):
        # actions must come in the following format:
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
        #     'asks' {
        #         source: {
        #             time_interval: {
        #                 'quantity': qty,
        #                 'price': dollar_per_kWh?
        #             }
        #         }
        #     }
        #
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

    async def check_read_flag(self, shared_list):
        """
        This method checks the read flag in a shared list.
        Parameters:
            Shared_list -> shared list object to check, assumes that element 0 is the flag and that flag can be
                            intepreted as boolean
            returns ->  Boolean
        """
        if shared_list[0]:
            return True
        else:
            return False

    async def read_action_values(self):
        """
        This method checks the action buffer flag and if the read flag is set, it reads the value in the buffer and stores
        them in self.actions

        """
        self.actions = []
        # check the action flag
        while True:
            flag = await self.check_read_flag(self.shared_list_action)
            print("Flag", flag)
            if flag:
                #read the buffer
                for e, item in enumerate(self.shared_list_action):
                    print(e, item)
                    self.actions.append(item)
                # self.actions = self.shared_list_action[1:]
                print('actions', self.actions)
                #reset the flag
                await self.write_flag(self.shared_list_action, False)
                break

    async def write_flag(self, shared_list, flag):
        """
        This method sets the flag
        Parameters:
            shared_list ->  shared list object to be modified
            flag -> boolean that indicates write 0 or 1. True sets 1
        """
        print(shared_list)

        if flag:
            shared_list[0] = 1
            print("Flag was set ")
        else:
            shared_list[0] = 0
            print("Flag was not set")

    async def write_observation_values(self, obs):
        """
        This method writes the values in the observations array to the observation buffer and then sets the flag for it

        """

        # obs will be an array
        # pack the values of the obs array into the shares list
        for e, item in enumerate(obs):
            print(e, item)
            self.shared_list_observation[e+1] = item

        #set the observation flat to written
        await self.write_flag(self.shared_list_observation,True)






