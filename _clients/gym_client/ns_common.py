import socketio

class NSDefault(socketio.AsyncClientNamespace):
    """
    This is the namespace for the gym_client
    """
    def __init__(self, gym_client):
        super().__init__(namespace='')
        self.gym_client = gym_client

class NSSimulation(socketio.AsyncClientNamespace):
    def __init__(self, gym_controller):
        super().__init__(namespace='/simulation')
        self.gym_controller = gym_controller


    async def on_connect(self):

        await self.gym_controller.register()

    async def on_get_remote_actions(self, message):
        """
        This event will trigger function in the gym client to query the agent for actions
        Args:
            message: observations from the Gym env.


        """
        await self.gym_controller.get_remote_actions(message)

    async def on_end_generation(self, message):
        print('notice me')