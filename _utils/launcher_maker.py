import os
import filecmp
import shutil
import json

class Maker:
    """
    This is the class that sets up the paths and arguments for the subprocess to run.
    The main idea is to make all pieces of the sim (market, server, patricipant, sim_controller) be calleable from the
    command line, and the launcher maker sets them up as variables for running as a subprocess.

    """
    def __init__(self, configs):
        self.configs=configs

    def get_config(self, config_json):
        """
        This method loads up the config file and outputs it as a dictionary object.
        Args:
            config_json: str path to configure file

        Returns:
            config: dictionary object based on the json config file
        """
        import json
        with open(config_json) as data_file:
            config = json.load(data_file)
        return config

    def get_server_configs(self):
        """
        This method gets the server host and port information from the config file.

        Returns:
            host:
            port:
        """
        host = self.configs['server']['host']
        port = str(self.configs['server']['port'])
        return host, port

    def make_sim_controller(self):
        """
        This method
        Returns:
            script_path: path to sim controller sio_client
            args: array of cli arguments
        """
        script_path = '_clients/sim_controller/sio_client.py'
        args = []
        host, port = self.get_server_configs()
        if host:
            args.append('--host=' + host)
        if port:
            args.append('--port=' + port)
        args.append('--config='+json.dumps(self.configs))
        return (script_path, args)

    def make_server(self):
        script_path = '_server/sio_server.py'

        args = []
        host, port = self.get_server_configs()

        if host:
            args.append('--host='+host)

        if port:
            args.append('--port='+port)

        return (script_path, args)

    def make_market(self):
        script_path = '_clients/markets/sio_client.py'
        args = []

        market_configs = self.configs['market']
        market_configs['timezone'] = self.configs['study']['timezone']
        host, port = self.get_server_configs()

        if host:
            args.append('--host=' + host)

        if port:
            args.append('--port=' + port)

        args.append('--configs=' + json.dumps(market_configs))
        return (script_path, args)

    def make_participant(self, participant_id, configs):
        script_path = '_clients/participants/sio_client.py'
        args = []

        participant_id = str(participant_id)
        if not participant_id:
            return False

        type = configs['type']
        host, port = self.get_server_configs()
        db_path = self.configs['study']['profiles_db_location']
        if 'sqlite:///' in db_path:
            db_path += participant_id + '.db'

        if not type:
            return False
        args.append(type)

        if participant_id:
            args.append('--id=' + participant_id)
            args.append('--market_id=' + self.configs['market']['id'])

        if host:
            args.append('--host=' + host)

        if port:
            args.append('--port=' + port)

        if db_path:
            args.append('--db_path=' + db_path)

        trader = configs['trader']
        if trader:
            args.append('--trader=' + json.dumps(trader))

        storage = configs['storage'] if 'storage' in configs else None
        if storage:
            storage_param_list = [
                str(storage['type']),
                str(storage['capacity']),
                str(storage['power']),
                str(storage['efficiency']),
                str(storage['monthly_sdr'])]
            storage_param_str = ",".join(storage_param_list)
            args.append('--storage=' + storage_param_str)

        if 'load' in configs and 'scale' in configs['load']:
            args.append('--load_scale=' + str(configs['load']['scale']))

        if 'generation' in configs and 'scale' in configs['generation']:
            args.append('--generation=' + str(configs['generation']['scale']))
        return (script_path, args)

    def make_gym(self):
        """

        Returns:
            script_path = str to sio client file for CLI initialization of the gym client
            args: array of strings that are
        """
        script_path = '_clients/gym_client/sio_client.py'
        args = []
        host, port = self.get_server_configs()
        if host:
            args.append('--host='+host)

        if port:
            args.append('--port='+port)
        return script_path, args

    def make_launch_list(self, make_server=True, make_market=True, make_sim_controller=True, make_participants=True,
                         make_gym=True):
        server = []
        market = []
        sim_controller = []
        participants = []
        gym = []

        if make_server:
            server.append(self.make_server())

        if make_market:
            market.append(self.make_market())

        if make_sim_controller:
            sim_controller.append(self.make_sim_controller())

        if make_participants:
            for p_id in self.configs['participants']:
                configs = self.configs['participants'][p_id]
                participants.append(self.make_participant(p_id, configs))

        if make_gym:
            gym.append(self.make_gym())

        return server, market, sim_controller, participants, gym