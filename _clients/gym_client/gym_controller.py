import socketio
from
class Controller:
    """
    This is the controller for the gym
    """

    def __init__(self, sio_client, configs, **kwargs):

        self.__client = sio_client
        self.id_list = []



    async def get_remote_actions(self, message):
        # this  TREX->gym->baselines
        #this needs to be
        participant_id = message.pop('participant_id')
        market_id = message.pop('market_id')

        #keep track of ids:
        # This TREX < -gym < -baselines

        action_dict = {}  # this should be an action dictionary, along with the agent that the actions should be routed to.
        #

        # actions needs to be of the form:
        actions = {
                      'participant_id': participant_id,
                      'market_id': market_id,
                  'actions': action_dict,

        }
        self.__client.emit('got_remote_actions', actions)


