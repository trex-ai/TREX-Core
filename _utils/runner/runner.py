class Runner:
    def __init__(self, config, resume=False, **kwargs):
        self.configs = self.__get_config(config, resume, **kwargs)
        self.__config_version_valid = bool(version.parse(self.configs['version']) >= version.parse("3.6.0"))

        # if not resume:
        #     r = tenacity.Retrying(
        #         wait=tenacity.wait_fixed(1))
        #     r.call(self.__make_sim_path)
    
    def __get_config(self, config_name:str, resume, **kwargs):
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

    # Give starting time for simulation
    def __get_start_time(self, generation):
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
    
    def __create_table(self, db_string, table):
        engine = create_engine(db_string)
        if not database_exists(engine.url):
            create_database(engine.url)
        table.create(engine, checkfirst=True)

    def __create_configs_table(self, db_string):
        table = sqlalchemy.Table(
            'configs',
            MetaData(),
            Column('id', sqlalchemy.Integer, primary_key=True),
            Column('data', sqlalchemy.JSON)
        )
        self.__create_table(db_string, table)

    def __create_metadata_table(self, db_string):
        table = sqlalchemy.Table(
            'metadata',
            MetaData(),
            Column('generation', sqlalchemy.Integer, primary_key=True),
            Column('data', sqlalchemy.JSON)
        )
        self.__create_table(db_string, table)

    def modify_config(simulation_type, **kwargs):
        if not self.__config_version_valid:
            return []
        
        seq = kwargs['seq'] if 'seq' in kwargs else 0

        config = json.loads(json.dumps(self.configs))
        default_port = int(self.configs['server']['port']) if self.configs['server']['port'] else 3000
        config['server']['port'] = default_port + seq

        config['study']['type'] = simulation_type
        learning_participants = [participant for participant in config['participants'] if
                                 'learning' in config['participants'][participant]['trader'] and
                                 config['participants'][participant]['trader']['learning']]

        if simulation_type == 'baseline':
            if isinstance(config['study']['start_datetime'], str):
                config['study']['generations'] = 2
            config['market']['id'] = type
            config['market']['save_transactions'] = True
            for participant in config['participants']:
                config['participants'][participant]['trader'].update({
                    'learning': False,
                    'type': 'baseline_agent'
                })

        if simulation_type == 'training':
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
        return config

    def make_launch_cli(self, config):
        from importlib import import_module
        from _utils.runner.make import sim_controller, server
        for key in config:
            import_module('_utils.runner.make', key)

        launch_list = []
        if 'server' in config:
            launch_list.append(server.make(config))

        if 'market' in config:
            launch_list.append(market.make(config))

        sim_controller.make(config)
        make.sim_controller.make(config)

        if make_participants:
            for p_id in configs['participants']:
                launch_list.append(make.participant.make(p_id, config))
        




    def run_subprocess(self, args: list, delay=0):
        import subprocess
        import time

        time.sleep(delay)

        try:
            subprocess.run(['venv/bin/python', args[0], *args[1]])
        except:
            subprocess.run(['venv/Scripts/python', args[0], *args[1]])
        finally:
            subprocess.run(['python', args[0], *args[1]])
        

    # def launch(self, simulations, skip_servers=False):
    #     if not self.__config_version_valid:
    #         print('CONFIG NOT COMPATIBLE')
    #         return
    #     from multiprocessing import Pool

    #     launch_list = []
    #     seq = 0
    #     for sim in simulations:
    #         launch_list.extend(self.make_one(**sim, seq=seq))
    #         seq += 1

    #     pool_size = len(launch_list)
    #     pool = Pool(pool_size)
    #     pool.map(self.launch_subprocess, launch_list)
    #     pool.close()