import tenacity
from _agent._utils.metrics import Metrics
import asyncio
from _agent._rewards import net_profit as reward
import numpy as np
import datetime


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

        last_settle = self.__participant['timing']['last_settle']
        hour = datetime.datetime.utcfromtimestamp(last_settle[0]).hour
        min = datetime.datetime.utcfromtimestamp(last_settle[0]).minute
        daytime = hour * 60 + min
        max_daytime = 24 * 60
        daytime_in_rad = 2 * np.pi * (daytime / max_daytime)
        day_sin = np.sin(daytime_in_rad)
        day_cos = np.cos(daytime_in_rad)

        next_settle = self.__participant['timing']['next_settle']
        generation, load = await self.__participant['read_profile'](next_settle)
        message = {
            'participant_id' : pid,
            'market_id': mid,
            'observations': {
                #observations stuff
                'next_settle_load_value': load,
                'next_settle_gen_value': generation,
                'last_settle_daytime_sin': day_sin,
                'last_settle_daytime_cos': day_cos

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

        # clear the next_actions variable and reset the wait_for_actions event
        self.next_actions.clear()
        self.wait_for_actions.clear()

        # get the next settle timing tuple from the participant
        next_settle = self.__participant['timing']['next_settle']

        # if the agent has storage, calculate max_charge and max_discharge values
        if 'storage' in self.__participant:
            storage_schedule = self.__participant['storage']['check_schedule'](next_settle)
            # storage_schedule = self.__participant['storage']['schedule'](next_settle)
            max_charge = storage_schedule[next_settle]['energy_potential'][1]
            max_discharge = storage_schedule[next_settle]['energy_potential'][0]

        # query observations from the remote agent
        observations = await self.get_observations()

        # Extract data from the observations received from the remote agent
        generation = observations['observations']['next_settle_gen_value']
        load = observations['observations']['next_settle_load_value']
        residual_load = load - generation
        residual_gen = -residual_load

        # Have the reward calculated
        rewards = await self._reward.calculate()
        # print('Reward from agent', rewards)
        observations['reward'] = rewards

        await self.__participant['emit']('get_remote_actions',
                                         data=observations,
                                         namespace='/simulation')

        # this is where we get actions: self.next_actions is where they will be deposited
        await self.wait_for_actions.wait()

        self.bid_price=self.next_actions['actions']['bids']['price']
        # print('bid price', self.bid_price)
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
                # print('we actually took an action', self.bid_price)
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
                self.metrics.track('bid_price', actions['bids'][str(next_settle)]['price']),
                self.metrics.track('bid_source', actions['bids'][str(next_settle)]['source']),
                self.metrics.track('bid_quantity', actions['bids'][str(next_settle)]['quantity']),
                self.metrics.track('next_settle_load', observations['observations']['next_settle_load_value']),
                self.metrics.track('next_settle_generation', observations['observations']['next_settle_gen_value']),
                self.metrics.track('reward', rewards)
                # self.metrics.track('available_quantity', final_residual_load)
            )
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



    def __init_metrics(self):
        import sqlalchemy
        '''
        Pretty self explanatory, this method resets the metric lists in 'agent_metrics' as well as zeroing the metrics dictionary. 
        '''
        self.metrics.add('timestamp', sqlalchemy.Integer)
        self.metrics.add('actions_dict', sqlalchemy.JSON)
        # Fixme: need to break up actions_dict into individual metrics
        # ask metrics:
        self.metrics.add('ask_price', sqlalchemy.Float)
        self.metrics.add('ask_quantity', sqlalchemy.Integer)
        self.metrics.add('ask_source', sqlalchemy.String)
        # bid metrics:
        self.metrics.add('bid_price', sqlalchemy.Float)
        self.metrics.add('bid_quantity', sqlalchemy.Integer)
        self.metrics.add('bid_source', sqlalchemy.String)
        # Load and gen metrics
        self.metrics.add('next_settle_load', sqlalchemy.Integer)
        self.metrics.add('next_settle_generation', sqlalchemy.Integer)
        # RL metrics:
        self.metrics.add('reward', sqlalchemy.Float)
        # self.metrics.add('available_quantity', sqlalchemy.Integer)

        if 'storage' in self.__participant:
            self.metrics.add('storage_soc', sqlalchemy.Float)
