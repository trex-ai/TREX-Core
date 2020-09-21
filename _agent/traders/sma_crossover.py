# from _clients.participants.participants import Residential
# 
"""This implements a SMA crossover trading strategy in the context of MicroFE
Moving averages are implemented as EMA inline with STC (Shaff Trend Cycle) for convenience. see https://www.tradingpedia.com/forex-trading-indicators/schaff-trend-cycle for more info on STC.

The basic premise is that a trading signal occurs when a short-term moving average (SMA) crosses through a long-term moving average (LMA). Buy signals occur when the SMA crosses above the LMA and a sell signal occurs during the opposite movement.

In finance, trade signals provide entry and exit points, likely at current market price. This is fine for BESS as the purpose is similar (arbitrage). 

For PV trading some sort of dynamic pricing method needs to be used.
BESS charge/discharge tracks PV price EMAs. For simplicity reasons, the dispatchable pool will not be used.
Instead, all bids and asks will go into the PV pool and BESS compensation will be used for discharging

15 step time windows
initialize with randomized prices for each window

trade with price in time step, then calculate per unit profit/cost for each time step
- if bid price too high, then the profit is equal to net metering (low),
- next time try lower price

calculate per unit cost for each time step
- if ask price too low, cost equal net metering (high)
- next time try higher price

"""
import tenacity
from _agent._utils.metrics import Metrics
import asyncio
from _utils import utils
from _utils import jkson as json
# from _agent._components import rewards
from _agent._rewards import unit_profit_and_cost as reward
import random

import sqlalchemy
from sqlalchemy import MetaData, Column
from _utils import db_utils
import databases
import ast
import collections
# import serialize

class EMA:
    def __init__(self, window_size):
        self.window_size = window_size
        self.count = 0
        self.last_average = 0

    def update(self, new_value):
        # approximate EMA
        self.count = min(self.count + 1, self.window_size)
        average = self.last_average + (new_value - self.last_average) / self.count
        self.last_average = average

    def reset(self):
        self.count = 0
        self.last_average = 0

class Trader:
    def __init__(self, bid_price, ask_price, **kwargs):
        # Some util stuffies
        self.__participant = kwargs['trader_fns']
        self.status = {
            'weights_loading': False,
            'weights_loaded': False,
            'weights_saving': False,
            'weights_saved': True
        }

        self.__db = {
            'path': '',
            'db': None
        }

        # Initialize the agent learning parameters for the agent (your choice)
        self.agent_data = {}
        scale_factor = 1
        self.sma_bid = EMA(23 * scale_factor) #23-period EMA for bid prices
        self.lma_bid = EMA(50 * scale_factor) #50-period EMA for bid prices

        self.sma_ask = EMA(23 * scale_factor)  # 23-period EMA for ask prices
        self.lma_ask = EMA(50 * scale_factor)  # 50-period EMA for ask prices

        self.buy_trigger = False
        self.sell_trigger = False


        # self.trade_signal = 0 # 1 if sma > lma else -1

        self.price_adj_step = 0.001 # next price is adjusted in half-cent steps

        self.bid_price = bid_price
        self.ask_price = ask_price

        # initialize price table
        self.bid_prices = self.__generate_price_table(bid_price, ask_price, 15)
        self.ask_prices = self.__generate_price_table(bid_price, ask_price, 15)

        # Initialize the metrics, whatever you
        # set learning and track_metrics flags
        self.learning = kwargs['learning'] if 'learning' in kwargs else False
        self._rewards = reward.Reward(self.__participant['timing'],
                                      self.__participant['ledger'],
                                      self.__participant['market_info'])


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
        if 'storage' in self.__participant:
            self.metrics.add('storage_soc', sqlalchemy.Float)

        # if self.battery:
        #     self.metrics.add('battery_action', sqlalchemy.Integer)
        #     self.metrics.add('state_of_charge', sqlalchemy.Float)

    def __generate_price_table(self, bid_price, ask_price, window_size):
        window_price = [random.randint(*sorted([bid_price*100, ask_price*100])) for i in range(int(1440/window_size))]
        # price_table = [[window_price[int(i/window_size)]/100, 0] for i in range(1440)]
        price_table = dict(zip(range(1440), [[window_price[int(i/window_size)]/100, 0] for i in range(1440)]))

        # print(price_table)
        # price_table = [{'price': window_price[int(i / window_size)] / 100, 'quantity': 0} for i in range(1440)]
        return price_table

    # Core Functions, learn and act, called from outside
    async def learn(self, **kwargs):
        if not self.learning:
            return

        rewards = await self._rewards.calculate()
        if rewards == None:
            return

        last_deliver = self.__participant['timing']['last_deliver']
        local_time = utils.timestamp_to_local(last_deliver[1], self.__participant['timing']['timezone'])
        # time_index = int(local_time.hour * 60 + local_time.minute) - 1
        time_index = int(local_time.hour * 60 + local_time.minute)

        # update price table
        # [unit_profit, unit_profit_diff, unit_cost, unit_cost_diff]
        unit_profit = rewards[0]
        unit_profit_diff = rewards[1]
        unit_cost = rewards[2]
        unit_cost_diff = rewards[3]

        last_ask_price = self.ask_prices[time_index][0]
        last_bid_price = self.bid_prices[time_index][0]

        if unit_profit_diff <= 0:
            self.ask_prices[time_index][0] = max(unit_profit, last_ask_price-self.price_adj_step)
        else:
            if unit_profit_diff - self.ask_prices[time_index][1] > 0:
                self.ask_prices[time_index][0] += self.price_adj_step
            else:
                self.ask_prices[time_index][0] -= self.price_adj_step
        self.ask_prices[time_index][1] = unit_profit_diff

        if unit_cost_diff >= 0:
            self.bid_prices[time_index][0] = min(unit_cost, last_bid_price + self.price_adj_step)
        else:
            if unit_cost_diff - self.bid_prices[time_index][1] > 0:
                self.bid_prices[time_index][0] -= self.price_adj_step
            else:
                self.bid_prices[time_index][0] += self.price_adj_step
        self.bid_prices[time_index][1] = unit_cost_diff

        # update EMA trackers
        self.sma_bid.update(self.bid_prices[time_index][0])
        self.lma_bid.update(self.bid_prices[time_index][0])

        self.sma_ask.update(self.ask_prices[time_index][0])
        self.lma_ask.update(self.ask_prices[time_index][0])

        # buy signal triggers when sma < lma (bid prices are rapidly dropping, therefore supply increasing)
        # sell signal triggers when sma > lma (ask prices are rapidly increasing, therefore demand increasing)

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


        # last_settle = self.__participant['timing']['current_round']
        next_settle = self.__participant['timing']['next_settle']
        # print(self.__participant['market_info'])
        grid_prices = self.__participant['market_info'][str(next_settle)]['grid']

        local_time = utils.timestamp_to_local(next_settle[1], self.__participant['timing']['timezone'])
        time_index = int(local_time.hour * 60 + local_time.minute)
        next_generation, next_load = await self.__participant['read_profile'](next_settle)
        next_residual_load = next_load - next_generation
        next_residual_gen = -next_residual_load

        # sell_trigger = False
        # buy_trigger = False

        if 'storage' in self.__participant:
            # battery actions are determined based on last_settle
            # last_settle = self.__participant['timing']['last_settle']
            current_round = self.__participant['timing']['current_round']
            # last_settle = self.__participant['timing']['current_round']
            ls_generation, ls_load = await self.__participant['read_profile'](current_round)
            ls_residual_load = ls_load - ls_generation
            ls_storage_schedule = await self.__participant['storage']['check_schedule'](current_round)
            ls_storage_scheduled = ls_storage_schedule[current_round]['energy_scheduled']
            settled = self.__participant['ledger'].get_settled_info(current_round)
            # if ls_storage_scheduled < 0:
            # #     # in discharge mode, trying to sell a bit more (res_load > 0)
            #     new_quantity = ls_storage_scheduled - last_settled['asks']['quantity']
            #     await self.__participant['storage']['schedule_energy'](new_quantity, last_settle)
            # print('--------------')
            # print(self.buy_trigger)
            # if self.buy_trigger:
            # if ls_storage_scheduled > 0:
            #    in charge mode, trying to buy a bit more (res_gen > 0)
            # print('-----------')
            # print(ls_storage_scheduled, self.buy_trigger, self.sell_trigger)
            if ls_storage_scheduled > 0:
                new_quantity = min(ls_storage_scheduled,
                                   max(0, settled['bids']['quantity'] - max(0, ls_residual_load)))

                # print('buy', ls_storage_scheduled, settled['bids']['quantity'], new_quantity)
                actions['bess'] = {str(current_round): new_quantity}
                # print(current_round, settled['bids']['quantity'], ls_residual_load, new_quantity)
            elif ls_storage_scheduled < 0:
                new_quantity = ls_storage_scheduled - settled['asks']['quantity']

                # print('sell', ls_storage_scheduled, settled['asks']['quantity'], new_quantity)
                actions['bess'] = {str(current_round): new_quantity}
                # print(current_round, settled['asks']['quantity'], new_quantity)



                # if last_settled['bids']['quantity'] > max(0, ls_residual_load) + new_quantity:

                # print(ls_storage_scheduled, ls_residual_load, new_quantity)


                # await self.__participant['storage']['schedule_energy'](new_quantity, last_settle)
                    # print(self.__participant['storage']['check_schedule'](last_settle)[last_settle]['energy_scheduled'])




            # in charge mode, trying to charge battery when res_load > 0
            # bid quantity is residual_load + max_charge
            # adjusted charge qty = settled - residual_load

            storage_schedule = await self.__participant['storage']['check_schedule'](next_settle)
            projected_soc = storage_schedule[next_settle]['projected_soc_end']
            # print(projected_soc)
            # buy signal triggers when sma < lma (bid prices are rapidly dropping, therefore supply increasing)
            # sell signal triggers when sma > lma (ask prices are rapidly increasing, therefore demand increasing)

        # residual_charge = 0
        # residual_discharge = 0
            # battery bids and asks are based on potentials charge/discharge capacities of next settle
            # if were lacking energy, get as much as possible out of battery

        # Logic if residence has both PV and bess:
        # We need a variable for energy currently being collected by solar panels, PV_supply
        # Also a variable for energy currently being consumed by residence, res_consumption
        # Still don't know how to come up with PV_supply and res_consumption though

        if PV_supply > res_consumption:
            ## all house needs fulfilled by PV
            PV_remaining = PV_supply - res_consumption
            if projected_soc <= 0.3:
                 if self.sma_ask.last_average > self.lma_ask.last_average:
                    ## sell all remaining PV to market
                    actions['asks'] = {
                        str(next_settle): {
                            'quantity': PV_remaining,
                            'source': 'solar',
                            'price': self.ask_prices[time_index][0]
                        }
                    }
                 else:
                    ## use remaining PV to charge battery
                    actions['bess'] = {str(current_round): PV_remaining}
                    self.buy_trigger = True
            if 0.3 < projected_soc < 0.9:
                 if self.sma_ask.last_average > self.lma_ask.last_average:
                    ## sell all remaining PV to market
                    actions['asks'] = {
                        str(next_settle): {
                            'quantity': PV_remaining,
                            'source': 'solar',
                            'price': self.ask_prices[time_index][0]
                        }
                    }
                    self.sell_trigger = True
                 else:
                    ## use remaining PV to charge battery
                    actions['bess'] = {str(current_round): PV_remaining}
                    self.buy_trigger = True
            if projected_soc >= 0.9:
                 ## sell all remaining PV to market
                 actions['asks'] = {
                    str(next_settle): {
                        'quantity': PV_remaining,
                        'source': 'solar',
                        'price': self.ask_prices[time_index][0]
                    }
                 }
                 if self.sma_ask.last_average > self.lma_ask.last_average:
                    self.sell_trigger = True
        else:
            ## all solar energy goes towards house needs
            remaining_house_needs = res_consumption - PV_supply
            if projected_soc <= 0.3:
                ## purchase remaining energy needs from market
                actions['bids'] = {
                    str(next_settle): {
                        'quantity': remaining_house_needs,
                        'source': 'solar',
                        'price': self.bid_prices[time_index][0]
                    }
                }
                if self.sma_ask.last_average < self.lma_ask.last_average:
                    self.buy_trigger = True
            if 0.3 < projected_soc < 0.9:
                if self.sma_ask.last_average > self.lma_ask.last_average:
                    ## battery fulfills remainder of energy needs
                    actions['bess'] = {str(current_round): -remaining_house_needs}
                    self.sell_trigger = True
                else:
                    ## purchase remaining energy needs from market
                    actions['bids'] = {
                        str(next_settle): {
                            'quantity': remaining_house_needs,
                            'source': 'solar',
                            'price': self.bid_prices[time_index][0]
                        }
                    }
                    self.buy_trigger = True
            if projected_soc >= 0.9:
                if self.sma_ask.last_average < self.lma_ask.last_average:
                    ## purchase remaining energy needs from market
                    actions['bids'] = {
                        str(next_settle): {
                            'quantity': remaining_house_needs,
                            'source': 'solar',
                            'price': self.bid_prices[time_index][0]
                        }
                    }
                else:
                    ## battery fulfills remainder of energy needs
                    actions['bess'] = {str(current_round): -remaining_house_needs}
                    self.sell_trigger = True

        if next_residual_load > 0:
            # sell_trigger = False
            # buy_trigger = False
            if 'storage' in self.__participant:
                # sell_trigger = bool(self.sma_ask.last_average > self.lma_ask.last_average)

                max_charge = storage_schedule[next_settle]['energy_potential'][1]
                max_discharge = storage_schedule[next_settle]['energy_potential'][0]

                # print(max_charge, max_discharge, projected_soc)

                effective_discharge = max(-max(0, next_residual_load), max_discharge)
                residual_discharge = max_discharge - effective_discharge

                if (self.sma_bid.last_average < self.lma_bid.last_average) and projected_soc < 0.9:
                # if self.sma_bid.last_average < self.lma_bid.last_average:
                    self.buy_trigger = True
                else:
                    self.buy_trigger = False

                # print('-----')
                # print(round(self.sma_bid.last_average, 2),
                #       round(self.lma_bid.last_average, 2),
                #       projected_soc,
                #       self.buy_trigger)

                if (self.sma_ask.last_average >= self.lma_ask.last_average) and projected_soc > 0.3:
                    self.sell_trigger=True
                else:
                    self.sell_trigger=False

                if self.buy_trigger and self.sell_trigger:
                    self.buy_trigger = False
                    self.sell_trigger = True


                if 'bess' in actions:
                    actions['bess'][str(next_settle)] = effective_discharge
                else:
                    actions['bess'] = {str(next_settle): effective_discharge}
            else:
                max_charge = 0
                effective_discharge = 0
                residual_discharge = 0

            if self.buy_trigger:
                actions['bids'] = {
                    str(next_settle): {
                        'quantity': next_residual_load + max_charge,
                        'source': 'solar',
                        'price': self.bid_prices[time_index][0]
                    }
                }
                if 'bess' in actions:
                    actions['bess'][str(next_settle)] = max_charge
                else:
                    actions['bess'] = {str(next_settle): max_charge}
            else:
                # self consume
                final_residual_load = next_residual_load + effective_discharge

                # if 'storage' in self.__participant:
                #     print(next_residual_load, final_residual_load, effective_discharge, residual_discharge)

                # if 'storage' in self.__participant:
                #     print(next_residual_load, effective_discharge)

                if final_residual_load > 0:
                    # bid for final residual load from market
                    # no battery charging, because it should be in discharge mode
                    actions['bids'] = {
                        str(next_settle): {
                            'quantity': final_residual_load,
                            'source': 'solar',
                            # 'price': max(grid_prices['sell_price'], min(grid_prices['buy_price'], self.bid_price_table[time_index][0]))
                            'price': self.bid_prices[time_index][0]
                        }
                    }
                else:
                    # there is excess discharge capacity to spare
                    if residual_discharge < 0 and self.sell_trigger:
                        actions['asks'] = {
                            str(next_settle): {
                                'quantity': abs(residual_discharge),
                                'source': 'solar',
                                # 'price': min(grid_prices['buy_price'], max(grid_prices['sell_price'], self.ask_price_table[time_index][0]))
                                'price': self.ask_prices[time_index][0]
                            }
                        }

        # if we have too much, cram as much as possible into battery
        elif next_residual_gen > 0:
            if 'storage' in self.__participant:
                # buy_trigger = bool(self.sma_bid.last_average < self.lma_bid.last_average)
                # buy_trigger = bool(self.sma_bid.last_average < self.lma_bid.last_average) and projected_soc < 0.8
                max_charge = storage_schedule[next_settle]['energy_potential'][1]
                effective_charge = min(next_residual_gen, max_charge)
                residual_charge = max_charge - effective_charge

                # if residual_charge and buy_trigger:
                #     actions['bids'] = {
                #         str(next_settle): {
                #             'quantity': residual_charge,
                #             'source': 'solar',
                #             'price': self.bid_prices[time_index][0]
                #         }
                #     }
                #     actions['bess'] = {str(next_settle): max_charge}
                # else:
                if abs(effective_charge):
                    actions['bess'] = {str(next_settle): effective_charge}
            else:
                effective_charge = 0

            # after self consumption into battery
            final_residual_gen = next_residual_gen - effective_charge
            if final_residual_gen > 0:
                actions['asks'] = {
                    str(next_settle): {
                        'quantity': final_residual_gen,
                        'source': 'solar',
                        # 'price': min(grid_prices['buy_price'], max(grid_prices['sell_price'], self.ask_price_table[time_index][0]))
                        'price': self.ask_prices[time_index][0]
                    }
                }

        if self.track_metrics:
            await asyncio.gather(
                self.metrics.track('timestamp', self.__participant['timing']['current_round'][1]),
                self.metrics.track('actions_dict', actions),
                self.metrics.track('next_settle_load', next_load),
                self.metrics.track('next_settle_generation', next_generation))
            if 'storage' in self.__participant:
                await self.metrics.track('storage_soc', projected_soc)


            await self.metrics.save(10000)
        return actions

    async def load_weights(self, db_path, generation, market_id, reset):
        if not self.__db['path']:
            import databases
            self.__db['path'] = db_path
            self.__db['db'] = databases.Database(db_path)

            if 'table' not in self.__db:
                weights_table_name = '_'.join(['weights', self.__participant['id']])
                weights_table = db_utils.get_table(db_path, weights_table_name)
                # print(weights_table_name, generation, type(weights_table))

                if weights_table is not None:
                    self.__db['table'] = weights_table
                else:
                    weights_table = sqlalchemy.Table(
                        weights_table_name,
                        MetaData(),
                        Column('generation', sqlalchemy.Integer, primary_key=True),
                        Column('bid_prices', sqlalchemy.String),
                        Column('ask_prices', sqlalchemy.String)
                    )
                    await db_utils.create_table(db_string=db_path, table_type='custom', custom_table=weights_table)
                    self.__db['table'] = weights_table
                await self.__db['db'].connect()

        if 'training' in market_id:
            return True

        if generation < 0 and 'validation' in market_id:
            return True

        db = self.__db['db']
        table = self.__db['table']
        query = table.select().where(table.c.generation == generation)
        async with db.transaction():
            row = await db.fetch_one(query)
            # print(row)
        if row is not None:
            self.bid_prices = ast.literal_eval(row['bid_prices'])
            self.ask_prices = ast.literal_eval(row['ask_prices'])
            return True
        return False

    async def save_weights(self, **kwargs):
        '''
        This method also saves the weights as well as setting and clearning flags associated with saving.

        Parameters:
            str : output path

        Returns:
            True
        '''

        if 'validation' in kwargs['market_id']:
            return True

        self.status['weights_saving'] = True
        self.status['weights_saved'] = False

        weights = [{
            'generation': kwargs['generation'],
            'bid_prices': str(self.bid_prices),
            'ask_prices': str(self.ask_prices)
        }]

        # print(self.__db['table'])

        asyncio.create_task(db_utils.dump_data(weights, kwargs['output_db'], self.__db['table']))
        self.status['weights_saving'] = False
        self.status['weights_saved'] = True
        return True


    async def reset(self, **kwargs):
        self.lma_bid.reset()
        self.lma_ask.reset()
        self.sma_bid.reset()
        self.sma_ask.reset()
        return True

    # async def __clip_price_tables(self, min_price, max_price):
    #     # clips prices between min and max of grid
    #     for idx in range(len(self.bid_price_table)):
    #         self.bid_price_table[idx][0] = min(max_price, max(min_price, self.bid_price_table[idx][0]))
    #         self.ask_price_table[idx][0] = min(max_price, max(min_price, self.ask_price_table[idx][0]))
