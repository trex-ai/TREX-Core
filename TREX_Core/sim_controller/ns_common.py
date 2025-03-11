import json
from pprint import pprint

# import socketio

class NSDefault():
    def __init__(self, controller):
        self.controller = controller

    async def process_message(self, message):
        # while True:
        #     msg = await msg_queue.get()
        topic_event = message['topic'].split('/')[-1]
        payload = message['payload']

        match topic_event:
            case 'market_online':
                await self.on_market_online(payload)
            case 'participant_joined':
                await self.on_participant_joined(payload)
            case 'end_turn':
                await self.on_end_turn(payload)
            case 'end_round':
                await self.on_end_round(payload)
            case 'participant_ready':
                await self.on_participant_ready(payload)
            case 'market_ready':
                await self.on_market_ready(payload)
            case 'participant_disconnected':
                await self.on_participant_disconnected(payload)
            case 'policy_server_ready':
                await self.on_policy_server_ready(payload)
            case 'sim_controller_status':
                await self.on_sim_controller_status()

    async def on_connect(self):
        await self.controller.register()

    async def on_participant_joined(self, message):
        participant_id = message
        await self.controller.participant_online(participant_id, True)

    async def on_participant_disconnected(self, message):
        print(message, 'PARTICIPANT LOST')
        participant_id = message
        await self.controller.participant_online(participant_id, False)

    async def on_participant_ready(self, message):
        message = json.loads(message)
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

    async def on_policy_server_ready(self, message):
        self.controller.status['policy_server_ready'] = True
        await self.controller.update_turn_status(message)

    async def on_sim_controller_status(self):
        pprint(self.controller.status)

    # async def on_end_simulation(self, message):
    #     raise SystemExit