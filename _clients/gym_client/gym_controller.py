import socketio

class Controller:
    """
    This is the controller for the gym
    """

    def __init__(self, sio_client, configs, **kwargs):

        self.__client = sio_client
        self.__config = configs

        # This is how the sim controller gets all the learning agents.
        # TODO: Create a way to filter the agents based on the config file
        # self.__learning_agents = [participant for participant in self.__config['participants'] if
        #                           'learning' in self.__config['participants'][participant]['trader'] and
        #                           self.__config['participants'][participant]['trader']['learning']]
