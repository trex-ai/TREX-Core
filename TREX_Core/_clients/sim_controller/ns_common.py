import socketio

class NSDefault(socketio.AsyncClientNamespace):
    def __init__(self, controller):
        super().__init__(namespace='')
        self.controller = controller

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

    # async def on_end_simulation(self, message):
    #     raise SystemExit