import socketio

class NSDefault(socketio.AsyncClientNamespace):
    """
    This is the namespace for the env_client
    """
    def __init__(self, env_client):
        super().__init__(namespace='')
        self.env_client = env_client

class NSSimulation(socketio.AsyncClientNamespace):
    def __init__(self, env_controller):
        super().__init__(namespace='/simulation')
        self.env_controller = env_controller


    async def on_connect(self):

        await self.env_controller.register()

    async def on_get_remote_actions(self, message):
        """
        This event will trigger function in the gym client to query the agent for actions
        Args:
            message: observations from the Gym env.
        """
        await self.env_controller.get_remote_actions(message)

    async def on_remote_agent_status(self, message):
        """

        :param message: the message is the market_id that the sim controller
        :return:
        """
        print('remote_agent_status')
        await self.env_controller.emit_go(message)

    async def on_end_generation(self, message):
        """
        This is now no longer the learn trigger. 
        On end generation, the gym agent will need to be reset 
        :param message:
        :return:
        """
        await self.env_controller.learn()
