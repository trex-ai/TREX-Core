import asyncio
import datetime
import sys
import json
import os
import time
import asyncio

import socketio
from sqlalchemy_utils import database_exists
import databases
import dataset
from _clients.sim_controller.training_controller import TrainingController
from _utils import utils, db_utils


class Controller:
    '''
    Sim controller takes over timing control from market
    in RT mode, the market will move onto the next round regardless of whether participants finish their actions or not
    in sim mode, market rounds will not start until all the participants have joined
    auction rounds (therefore time) will not advance until all participants have completed all of their actions

    in order to do this, the sim controller needs to know the following things about the simulation
    this information can be obtained from the config json file

    1. sim start time
    2. number of participants, as well as their IDs

    The sim controller has special permission to see when participants join the market
    '''
    # Intialize client related data
    def __init__(self, sio_client, configs, **kwargs):
        self.__client = sio_client
        self.__config = configs

        self.__learning_agents = [participant for participant in self.__config['participants'] if
                                 'learning' in self.__config['participants'][participant]['trader'] and
                                 self.__config['participants'][participant]['trader']['learning']]

        self.__static_agents = [participant for participant in self.__config['participants'] if
                                'learning' not in self.__config['participants'][participant]['trader'] or
                                 not self.__config['participants'][participant]['trader']['learning']]
        self.__participants = {}
        self.__turn_control = {
            'total': None,
            'online': 0,
            'ready': 0,
            'weights_loaded': 0,
            # 'weights_saved': 0,
            'ended': 0,
        }

        self.__generations = self.__config['study']['generations'] - 1
        self.__generation = self.set_initial_generation()

        self.__start_time = self.get_start_time()
        self.__time = self.__start_time

        self.__current_step = 0
        self.__end_step = int(self.__config['study']['days'] * 1440) + 1

        self.__time_step_s = 60
        self.make_participant_tracker()

        self.timer_start = datetime.datetime.now().timestamp()
        self.timer_end = 0

        self.status = {
            'monitor_timeout': 5,
            'registered_on_server': False,
            'market_id': self.__config['market']['id'],
            'sim_started': False,
            'sim_ended': False,
            'generation_ended': False,
            'current_step': self.__current_step,
            'last_step_clock': None,
            'running_generations': 0,
            'market_online': False,
            'market_ready': True,
            'sim_interrupted': False,
            'participants': self.__participants,
            'learning_agents': self.__learning_agents,
            'participants_online': False,
            'participants_ready': True,
            'participants_weights_loaded': False,
            # 'participants_weights_saved': True,
            'turn_control': self.__turn_control,
            'market_turn_end': False,
        }
<<<<<<< HEAD

        if 'remote_agent' in self.__config:
            self.status['remote_agent_ready'] = False

=======
>>>>>>> master
        self.training_controller = TrainingController(self.__config, self.status)

    async def delay(self, s):
        '''This function delays the sim by s seconds using the client sleep method so as not to interrupt the thread control. 

        Params: 
            int or float : number of seconds to 
        '''
        await self.__client.sleep(s)

    def __get_metadata(self, generation):
        if generation > self.__generations:
            return None

        db_string = self.__config['study']['output_database']
        db = dataset.connect(db_string)
        md_table = db['metadata']
        metadata = md_table.find_one(generation=generation)
        return metadata['data']

    def get_start_time(self):
        metadata = self.__get_metadata(self.__generation)
        if metadata:
            return metadata['start_timestamp']

        import pytz
        from dateutil.parser import parse as timeparse
        tz_str = self.__config['study']['timezone']
        dt_str = self.__config['study']['start_datetime']

        start_time = pytz.timezone(tz_str).localize(timeparse(dt_str))
        return int(start_time.timestamp())

    # Initialize data for participant turns
    def make_participant_tracker(self):
        self.__turn_control['total'] = len(self.__config['participants'])
        for participant_id in list(self.__config['participants']):
            self.__participants[participant_id] = {
                'online': False,
                'turn_end': False,
                'ready': False,
                'weights_loaded': False,
                # 'weights_saved': False
            }

    # Set intial generation data folder
    # @tenacity.retry(wait=tenacity.wait_fixed(5)+tenacity.wait_random(0, 5))
    #TODO: update this to check all dbs
    def set_initial_generation(self):
        db_string = self.__config['study']['output_database']
        if not database_exists(db_string):
            return 0

        #TODO: rewrite generation detection for resume
        if self.__config['study']['resume']:
            pass
            # return 0

        return 0

        # for generation in range(self.__config['study']['generations']):
        #     metadata = self.__get_metadata(generation)
        #     if self.__config['market']['id'] not in metadata:
        #         market_table_name = str(generation) + '_' + self.__config['market']['id']
        #         metrics_table_name = market_table_name + '_metrics'
        #         db_utils.drop_table(db_string, market_table_name)
        #         db_utils.drop_table(db_string, metrics_table_name)
        #         return generation
        # return self.__generations

    # Register client in server
    async def register(self):
        client_data = {
            'type': ('sim_controller', ''),
            'id': '',
            'market_id': self.__config['market']['id']
        }
        await self.__client.emit('register', client_data, namespace='/simulation', callback=self.register_success)

    # If client has not connected, retry registration
    async def register_success(self, success):
        await self.delay(utils.secure_random.random() * 10)
        if not success:
            await self.register()
        self.status['registered_on_server'] = True

    # Track ending of turns
    async def update_turn_status(self, participant_id):
        if participant_id in self.__participants:
            self.__participants[participant_id]['turn_end'] = True
            self.__turn_control['ended'] += 1

        if self.__turn_control['ended'] < self.__turn_control['total']:
            return
        if self.status['market_turn_end']:
            await self.__advance_turn()

    def __reset_turn_trackers(self):
        self.status['market_turn_end'] = False
        for participant_id in self.__participants:
            self.__participants[participant_id]['turn_end'] = False
        self.__turn_control['ended'] = 0

    async def __advance_turn(self):
        # Once all participants have gone through their turn,
        # and market is ready
        # reset turn trackers
        # take next step
        self.__reset_turn_trackers()
        self.__time += self.__time_step_s
        await self.step()

    # Reset turn count
    async def market_turn_end(self):
        self.status['market_turn_end'] = True

    # Update tracker when participant is active
    async def participant_status(self, participant_id, status, condition):
        if participant_id in self.__participants:
            last_condition = self.__participants[participant_id][status]
            if condition == last_condition:
                return
            self.__participants[participant_id][status] = condition
            if condition:
                self.__turn_control[status] = min(self.__turn_control['total'], self.__turn_control[status] + 1)
            else:
                self.__turn_control[status] = max(0, self.__turn_control[status] - 1)
        if self.__turn_control[status] < self.__turn_control['total']:
            self.status['participants_' + status] = False
        else:
            self.status['participants_' + status] = True

    # Update tracker when participant is active
    async def participant_online(self, participant_id, online):
        if participant_id in self.__participants:
            self.__participants[participant_id]['online'] = online
            if online:
                self.__turn_control['online'] = min(self.__turn_control['total'], self.__turn_control['online'] + 1)
            else:
                self.__turn_control['online'] = max(0, self.__turn_control['total'] - 1)
                self.status['sim_interrupted'] = True
                self.status['sim_started'] = False
        if self.__turn_control['online'] < self.__turn_control['total']:
            self.status['participants_online'] = False
        else:
            self.status['participants_online'] = True

    async def monitor(self):
        while True:
            await self.delay(self.status['monitor_timeout'])
            if not self.status['registered_on_server']:
                continue

            if not self.status['generation_ended'] and self.status['last_step_clock'] and time.time() - self.status['last_step_clock'] > 600:
                self.status['last_step_clock'] = time.time()
                print(self.status)

                #TODO: One of the most likely scensarios for sim to get stuck is that a participant
                # disconnects before an action is taken for some reason, so that the turn tracker cannot advance
                # In the event that this happens, a set of checks need to be performed to resume where the agent abruptly died.

                message = {
                    'time': self.__time,
                    'duration': self.__time_step_s,
                    'update': False
                }
                await self.__client.emit('start_round', message, namespace='/simulation')

            if self.status['sim_started']:
                continue

            if not self.status['market_online']:
                await self.__client.emit('is_market_online', namespace='/simulation')
                continue

            if 'remote_agent_ready' in self.status and not self.status['remote_agent_ready']:
                message = {
                    'market_id': self.__config['market']['id']
                }
                await self.__client.emit(event='remote_agent_status',
                                         data=message,
                                         namespace='/simulation')
                continue

            if not self.status['participants_online']:
                await self.__client.emit('re_register_participant', namespace='/simulation')
                continue

            if not self.status['market_ready']:
                continue

            # await self.update_sim_paths()

            if self.status['sim_ended']:
                continue
            # if self.__generation > self.__generations and not self.status['sim_ended']:
            #     self.status['sim_ended'] = True
            #     continue

            if self.__config['study']['type'] == 'training':
                curriculum = self.training_controller.load_curriculum(str(self.__generation))
                if curriculum:
                    await self.__client.emit('update_curriculum', curriculum, namespace='/simulation')

            if not self.status['participants_ready']:
                continue

            #for now, only load weights for validation
            if self.__config['study']['type'] == 'validation':
                market_id = 'training'
                if not self.status['participants_weights_loaded']:
                    db = dataset.connect(self.__config['study']['output_database'])
                    for participant_id in self.__participants:
                        await self.__load_weights(db, self.__generation, market_id, participant_id)
                    continue

            if self.status['sim_interrupted']:
                print('drop drop')
                if self.__turn_control['total'] - self.__turn_control['online'] > 1:
                    self.__current_step = 0
                else:
                    await self.__client.emit('re_register_participant', namespace='/simulation')
                self.status['sim_interrupted'] = False
                continue

            self.status['sim_started'] = True
            self.status['monitor_timeout'] = 5
            await self.step()

    async def __load_weights(self, db, generation, market_id, participant_id):
        # db_string = self.__config['study']['output_database']
        # db = dataset.connect(db_string)
        weights_table_name = '_'.join((str(generation), market_id, 'weights', participant_id))
        # weights_table = db[weights_table_name]
        # weights = weights_table.find_one(generation=generation)
        if weights_table_name not in db.tables:
            self.status['monitor_timeout'] = 30
            return

        message = {
            'participant_id': participant_id,
            'db_path': self.__config['study']['output_database'],
            'market_id': market_id,
            'generation': generation
        }
        await self.__client.emit('load_weights', message, namespace='/simulation')

    async def __print_step_time(self):
        if self.__current_step % 1440 == 0:
            # Print time information for time step/ expected runtime
            end = datetime.datetime.now().timestamp()
            step_time = end - self.timer_start
            eta_s = round((self.__end_step - self.__current_step) / 1440 * step_time)
            print(self.__config['market']['id'],
                  ', generation', self.__generation, '/', self.__generations,
                  ', day', int(self.__current_step / 1440), '/', int((self.__end_step - 1) / 1440))
            print('step time:', round(step_time, 0), 's', ', ETA:', str(datetime.timedelta(seconds=eta_s)))
            self.timer_start = datetime.datetime.now().timestamp()

    async def step(self):
        self.status['last_step_clock'] = time.time()
        # if self.status['sim_ended']:
        #     print('end_simulation', self.__generation, self.__generations)
        #     await self.__client.emit('end_simulation', namespace='/simulation')
        #     await self.delay(1)
        #     raise SystemExit

        if not self.status['sim_started']:
            return

        # Beginning new generation
        if self.__current_step == 0:
            print('STARTING SIMULATION')
            message = {
                'generation': self.__generation,
                'db_string': self.__config['study']['output_database'],
                # 'input_path': self.status['input_path'],
                # 'output_path': self.status['output_path'],
                'market_id': self.__config['market']['id'],
            }
            await self.__client.emit('start_generation', message, namespace='/simulation')
            self.status['generation_ended'] = False

        # Beginning new time step
        if self.__current_step <= self.__end_step:
            await self.__print_step_time()
            self.__current_step += 1

            message = {
                'time': self.__time,
                'duration': self.__time_step_s,
                'update': True
            }
            await self.__client.emit('start_round', message, namespace='/simulation')
        # end of generation
        elif self.__current_step == self.__end_step + 1:
            self.__turn_control.update({
                'ready': 0,
                'weights_loaded': 0,
                # 'weights_saved': 0
            })
            for participant_id in self.__participants:
                self.__participants[participant_id].update({
                    'ready': False,
                    'weights_loaded': False,
                # 'weights_saved': False
            })
            self.status['participants_ready'] = False
            # self.status['participants_weights_saved'] = False
            self.status['participants_weights_loaded'] = False

            self.status['generation_ended'] = True
            # await db_utils.update_metadata(self.__config['study']['output_database'],
            #                                self.__generation,
            #                                {self.__config['market']['id']: True})

            # end simulation if the final generation is done, else reset step and stuff
            if self.__generation <= self.__generations:
                print('generation', self.__generation, 'complete')
                self.__generation += 1
                self.status['running_generations'] += 1
                self.__current_step = 0
                self.__start_time = self.get_start_time()
                self.__time = self.__start_time
                self.status['sim_started'] = False
                self.status['market_ready'] = False
                if 'remote_agent_ready' in self.status:
                    self.status['remote_agent_ready'] = False

            message = {
                # 'output_path': self.status['output_path'],
                'db_path': self.__config['study']['output_database'],
                'generation': self.__generation - 1,
                'market_id': self.__config['market']['id']
            }
            await self.__client.emit('end_generation', message, namespace='/simulation')

            if self.__generation > self.__generations:
                self.status['sim_ended'] = True
                # if self.status['sim_ended']:
                print('end_simulation', self.__generation-1, self.__generations)
                await self.__client.emit('end_simulation', namespace='/simulation')
                await self.delay(1)
                sys.exit()

class NSMarket(socketio.AsyncClientNamespace):

    def __init__(self, controller):
        super().__init__(namespace='/market')
        self.controller = controller

    # async def on_connect(self):
    #     print('connected to market')
    #     # await self.controller.register()

    # async def on_disconnect(self):
    #     print('disconnected from market')

    # async def on_register(self):
    #     print('participant registered')

    # # async def on_end_round(self, message):
    # #     await self.controller.step()

class NSSimulation(socketio.AsyncClientNamespace):
    def __init__(self, controller):
        super().__init__(namespace='/simulation')
        self.controller = controller

    async def on_connect(self):
        await self.controller.register()

    # async def on_disconnect(self):
    #   print('disconnected from simulation')


    #
    # async def on_participant_weights_saved(self, message):
    #     for participant_id in message:
    #         await self.controller.participant_status(participant_id, 'weights_saved', message[participant_id])


    async def on_participant_joined(self, message):
        participant_id = message
        await self.controller.participant_online(participant_id, True)

    async def on_participant_disconnected(self, message):
        print(message, 'PARTICIPANT LOST')
        participant_id = message
        await self.controller.participant_online(participant_id, False)

    async def on_participant_ready(self, message):
        for participant_id in message:
            await self.controller.participant_status(participant_id, 'ready', message[participant_id])

    async def on_participant_weights_loaded(self, message):
        for participant_id in message:
            await self.controller.participant_status(participant_id, 'weights_loaded', message[participant_id])

    # send by individual participants
    async def on_end_turn(self, message):
        await self.controller.update_turn_status(message)

    # sent by the market
    async def on_end_round(self, message):
        await self.controller.market_turn_end()
        await self.controller.update_turn_status(message)

    async def on_market_online(self, message):
        self.controller.status['market_online'] = True

    async def on_market_ready(self, message):
        self.controller.status['market_ready'] = True

    async def on_remote_agent_ready(self):
        if 'remote_agent_ready' in self.controller.status:
            self.controller.status['remote_agent_ready'] = True
    # async def on_end_simulation(self, message):
    #     raise SystemExit

