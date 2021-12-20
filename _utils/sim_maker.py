import os
import filecmp
import shutil
import commentjson
import gzip
import random
from _utils import launcher_maker
from _utils import db_utils
from _utils import jkson as json
import tenacity
import sqlalchemy
from sqlalchemy import create_engine, MetaData, Column
from sqlalchemy_utils import database_exists, create_database, drop_database
import dataset
from packaging import version
import time

class Maker:
    def __init__(self, config, resume=False, **kwargs):
        """
        This constructor gets the configs using __get_config(config) and puts it into self.config

        Args:
            config: a string to the config file
            resume: Boolean
            **kwargs:
        """
        self.configs = self.__get_config(config, resume, **kwargs)
        self.__config_version_valid = bool(version.parse(self.configs['version']) >= version.parse("3.6.0"))

        if not resume:
            r = tenacity.Retrying(
                wait=tenacity.wait_fixed(1))
            r.call(self.__make_sim_path)

    def __get_config(self, config_name:str, resume, **kwargs):
        """
        This method loads the configs from the json. In addition, this method sets up the database engine and in the
        event that resume is true, it links up to the existing database and pulls the config from there.

        This method does:
            1. Loads the json config file based on the config name that is passed into main.
            2. Sets up the database engine based on the config
            3. If resume is on, query the database for the config that is stored in it
            4. If there is no database, create the database based on the config
            5. Create the configs table by calling __create_configs_table()
            6. Insert the config into the database
        Return: config

        Args:
            config_name:
            resume:
            **kwargs:

        Returns:

        """
        # TODO Nov 30, 2021; this may also need to be modified to have the path to the trex-core directory attached to
        # it
        config_file = '_configs/' + config_name + '.json'
        with open(config_file) as f:
            config = commentjson.load(f)

        if 'name' in config['study'] and config['study']['name']:
            study_name = config['study']['name'].replace(' ', '_')
        else:
            study_name = config_name
        db_string = config['study']['output_db_location'] + '/' + study_name
        engine = create_engine(db_string)

        if resume:
            if 'db_string' in kwargs:
                db_string = kwargs['db_string']
            # look for existing db in db. if one exists, return it
            if database_exists(db_string):
                if engine.dialect.has_table(engine, 'configs'):
                    db = dataset.connect(db_string)
                    configs_table = db['configs']
                    configs = configs_table.find_one(id=0)['data']
                    configs['study']['resume'] = resume
                    return configs

        config['study']['name'] = study_name
        if 'output_database' not in config['study'] or not config['study']['output_database']:
            config['study']['output_database'] = db_string

        if database_exists(db_string):
            drop_database(db_string)
        if not database_exists(db_string):
            db_utils.create_db(db_string)

        self.__create_configs_table(db_string)
        db = dataset.connect(db_string)
        configs_table = db['configs']
        configs_table.insert({'id': 0, 'data': config})

        config['study']['resume'] = resume
        return config

    def __make_sim_path(self):
        """
        This method creates the directory path in the _simultions/ directory and then creates the directories.

        What this does:
            1. Composes the simulation path from the config parameters
            2. Checkes if this path already exists
                a. If it does, delete everything
            3. Make the directory specified by the simulation path

        Returns:

        """
        # TODO: Nov 30 2021; This may need to be also altered to have the simulation save properly
        output_path = self.configs['study']['sim_root'] + '_simulations/' + self.configs['study']['name'] + '/'
        print(output_path)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)

    # Give starting time for simulation
    def __get_start_time(self, generation):
        """
        This method gives the starting timestep for the simulation

            What it does:
                1. Grabs the startime and timezone information from the configs
                2. If the startdatetime is a single value, return that value
                3. If the start time is a tuple (start, end) then return the start
                4.If the study is sequential, then calculate the startime from the config parameters including the
                interval and return the startime.
        Args:
            generation:

        Returns:

        """
        import pytz
        import math
        from dateutil.parser import parse as timeparse
        #  TODO: NEED TO CHECK ALL DATABASES TO ENSURE THAT THE TIME RANGE ARE GOOD
        start_datetime = self.configs['study']['start_datetime']
        start_timezone = self.configs['study']['timezone']

        # If start_datetime is a single time, set that as start time
        if isinstance(start_datetime, str):
            start_time = pytz.timezone(start_timezone).localize(timeparse(start_datetime))
            return int(start_time.timestamp())

        # If start_datetime is formatted as a time step with beginning and end, choose either of these as a start time
        # If sequential is set then the startime will 
        if isinstance(start_datetime, (list, tuple)):
            if len(start_datetime) == 2:
                start_time_s = int(pytz.timezone(start_timezone).localize(timeparse(start_datetime[0])).timestamp())
                start_time_e = int(pytz.timezone(start_timezone).localize(timeparse(start_datetime[1])).timestamp())
                # This is the sequential startime code 
                if 'start_datetime_sequence' in self.configs['study']:
                    if self.configs['study']['start_datetime_sequence'] == 'sequential':
                        interval = int((start_time_e-start_time_s) / self.configs['study']['generations']/60)*60
                        start_time = range(start_time_s, start_time_e, interval)[generation]
                        return start_time
                start_time = random.choice(range(start_time_s, start_time_e, 60))
                return start_time
            else:
                 if 'start_datetime_sequence' in self.configs['study']:
                    if self.configs['study']['start_datetime_sequence'] == 'sequential':
                        multiplier = math.ceil(self.configs['study']['generations']/len(start_datetime))
                        start_time_readable = start_datetime*multiplier[generation]
                        start_time = pytz.timezone(start_timezone).localize(timeparse(start_time_readable))
                        return start_time
                 start_time = pytz.timezone(start_timezone).localize(timeparse(random.choice(start_datetime)))
                 return int(start_time.timestamp())

    def __make_sim_internal_directories(self, config=None):
        """
        This method sets up the database directories in _simulations as well as setting up the metadata tables in the
        database under a metadata table.

        This function does:
            1. Creates a directory for the simulation in the _simulations directory
            2. Connects to the database
            3. Checks if there is a metadata table already there.
            4. For each genration, creates metadata tables that contain the start and end tumestamp values

        Args:
            config:

        Returns:

        """
        if not config:
            config = self.configs
        # make sim directories and shared settings files
        sim_path = self.configs['study']['sim_root'] + '_simulations/' + config['study']['name'] + '/'
        if not os.path.exists(sim_path):
            os.mkdir(sim_path)

        engine = create_engine(self.configs['study']['output_database'])
        if not engine.dialect.has_table(engine, 'metadata'):
            self.__create_metadata_table(self.configs['study']['output_database'])
        db = dataset.connect(self.configs['study']['output_database'])
        metadata_table = db['metadata']
        for generation in range(config['study']['generations']):
            # check if metadata is in table
            # if not, then add to table
            if not metadata_table.find_one(generation=generation):
                start_time = self.__get_start_time(generation)
                metadata = {
                    'start_timestamp': start_time,
                    'end_timestamp': int(start_time + self.configs['study']['days'] * 1440)
                }
                metadata_table.insert(dict(generation=generation, data=metadata))

    def __create_table(self, db_string, table):
        """
        This method sends the table that gets passed into the method to the database.

        What it does:
            1. Connects to the database
            2. Checks that the database exists
            3. Creates the table in the database

        Args:
            db_string:
            table:

        Returns:

        """
        engine = create_engine(db_string)
        if not database_exists(engine.url):
            create_database(engine.url)
        table.create(engine, checkfirst=True)

    def __create_configs_table(self, db_string):
        """
        This method sets up the configs table and then calls __create_table() to create the table in the database.
        Args:
            db_string:

        Returns:

        """
        table = sqlalchemy.Table(
            'configs',
            MetaData(),
            Column('id', sqlalchemy.Integer, primary_key=True),
            Column('data', sqlalchemy.JSON)
        )
        self.__create_table(db_string, table)

    def __create_metadata_table(self, db_string):
        """
        This method sets up the metadata information and then calls the __create_table()

        Args:
            db_string:

        Returns:

        """
        table = sqlalchemy.Table(
            'metadata',
            MetaData(),
            Column('generation', sqlalchemy.Integer, primary_key=True),
            Column('data', sqlalchemy.JSON)
        )
        self.__create_table(db_string, table)
    
    def make_one(self, type:str, mode:str='', seq=0, skip_server=False, **kwargs):
        """
        This method sets up the launch sequence based on the type of simulation that you are trying to run.

        This method does:
            1. Sets up the config based on your simulation type
            2. Runs __make_sim_internal_directories()
            3. Runs launcher_maker(Maker) this passes the config to the launcher_maker.Maker object.
            4. Runs launcher_maker.make_launch_list()


        Args:
            type:
            mode:
            seq:
            skip_server:
            **kwargs:

        Returns:
            launch_sequence: array of cli arguments for python subprocess

        """
        if not self.__config_version_valid:
            return []
        config = json.loads(json.dumps(self.configs))
        default_port = int(self.configs['server']['port']) if self.configs['server']['port'] else 3000
        config['server']['port'] = default_port + seq
        config['study']['type'] = type
        learning_participants = [participant for participant in config['participants'] if
                                 'learning' in config['participants'][participant]['trader'] and
                                 config['participants'][participant]['trader']['learning']]

        if type == 'baseline':
            if isinstance(config['study']['start_datetime'], str):
                config['study']['generations'] = 2
            config['market']['id'] = type
            config['market']['save_transactions'] = True
            for participant in config['participants']:
                config['participants'][participant]['trader'].update({
                    'learning': False,
                    'type': 'baseline_agent'
                })

        if type == 'training':
            config['market']['id'] = type
            config['market']['save_transactions'] = True

            if 'target' in kwargs:
                if not kwargs['target'] in config['participants']:
                    return []

                config['market']['id'] += '-' + kwargs['target']
                for participant in learning_participants:
                    config['participants'][participant]['trader']['learning'] = False
                config['participants'][kwargs['target']]['trader']['learning'] = True
            else:
                for participant in learning_participants:
                    config['participants'][participant]['trader']['learning'] = True

        if type == 'validation':
            config['market']['id'] = type
            config['market']['save_transactions'] = True

            for participant in config['participants']:
                config['participants'][participant]['trader']['learning'] = False

        self.__make_sim_internal_directories()
        lmaker = launcher_maker.Maker(config)
        # TODO: Nov 24 This is where you can modify the launch list for debugging
        server, market, sim_controller, participants, gym = lmaker.make_launch_list(make_participants=True, make_gym=True,
                                                                                    make_market=True)
        launch_sequence = market + sim_controller + participants + gym
        print("Launch list", launch_sequence)
        if not skip_server:
            launch_sequence = server + launch_sequence
        return launch_sequence

    def launch_subprocess(self, args: list, delay=0):
        """
        This method takes the arguments from args (ouput of launcher_maker.make_launch_list()) and gets python to run them.

        This method does:
            1. Checks if the string passed in ends in a .py
            2. Runs this using the env/bin/python
            3. If that does not work, trys venv/Scripts/Python
            4. Finally resorts to the system python

        Args:
            args:
            delay:

        Returns:

        """
        time.sleep(delay)
        # TODO: Nov 30 2021; added this to try to get remote launch
        path_to_trex = str(Path('C:/source/Trex-Core/'))
        path_to_venv = path_to_trex + str(Path('/venv/Scripts/python'))
        path_to_trex_main = path_to_trex + str(Path('/main.py'))

        import subprocess
        extension = args[0].split('.')[1]
        is_python = True if extension == 'py' else False

        if is_python:
            try:
                # TODO: Nov 30, 2021; added this to try to get remote launch
                print('path in sim maker', path_to_venv)
                subprocess.run([path_to_venv, args[0], *args[1]])  # arg[0] is the server
                # subprocess.run(['env/bin/python', args[0], *args[1]]) # arg[0] is the server
            except:
                subprocess.run(['venv/Scripts/python', args[0], *args[1]])
                # subprocess.run(['venv/Scripts/python', args[0], *args[1]])
            finally:
                subprocess.run(['python', args[0], *args[1]])
        else:
            subprocess.run([args[0], *args[1]])

    def launch(self, simulations, skip_servers=False):
        """
        This method sets up the multiprocess pool for cli python subprocess and maps each pool with its own subprocess.
        This method does:
            1. For each sim in the simulation parameter
                a. Call make_one and add it to a list
            2. Set up the multiprocess pool
            3. Map launch_subprocess with the output of make_one to each process in the pool

        Args:
            simulations:
            skip_servers:

        Returns:

        """
        if not self.__config_version_valid:
            print('CONFIG NOT COMPATIBLE you noob!!')
            return
        from multiprocessing import Pool

        launch_list = []
        seq = 0
        for sim in simulations:
            launch_list.extend(self.make_one(**sim, seq=seq))
            seq += 1

        pool_size = len(launch_list)
        pool = Pool(pool_size)
        pool.map(self.launch_subprocess, launch_list)
        pool.close()
