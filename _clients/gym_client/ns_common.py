import socketio

class NSDefault(socketio.AsyncClientNamespace):
    """
    This is the namespace for the gym_client
    """
    def __init__(self, gym_client):
        super().__init__(namespace='')
        self.gym_client = gym_client