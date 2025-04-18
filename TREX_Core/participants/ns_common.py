# import socketio
import json
import asyncio
import numpy as np

class NSDefault:
    def __init__(self, participant):
        # super().__init__(namespace='')
        self.participant = participant

    async def process_message(self, message):
        topic_event = message['topic'].split('/')[-1]
        payload = message['payload']

        match topic_event:
            # market related events
            case 'market_info':
                await self.on_update_market_info(payload)
            case 'start_round':
                await self.on_start_round(payload)
            case 'ask_ack':
                await self.on_ask_success(payload)
            case 'bid_ack':
                await self.on_bid_success(payload)
            case 'settled':
                # print('settled?')
                await self.on_settled(payload)
            case 'extra_transaction':
                await self.on_return_extra_transactions(payload)
            # simulation related events
            case 'is_participant_joined':
                await self.on_is_participant_joined(payload)
            case 'start_episode':
                await self.on_start_episode(payload)
            case 'end_episode':
                await self.on_end_episode(payload)
            case 'end_simulation':
                await self.on_end_simulation()
            case 'get_actions_return':
                await self.on_get_actions_return(payload)
            case 'get_metadata_return':
                await self.on_get_metadata_return(payload)

    async def on_connect(self):
        # print('connected')
        await self.participant.open_profile_db()
        self.participant.server_online = True
        await self.participant.join_market()

    # async def on_disconnect(self):
    def on_disconnect(self):
        self.participant.server_online = False
        self.participant.busy = True
        # print("participant disconnected")

    async def on_update_market_info(self, payload):
        client_data = json.loads(payload)
        if client_data['id'] == self.participant.market_id:
            self.participant.market_sid = client_data['sid']
            self.participant.timezone = client_data['timezone']
            self.participant.market_connected = True
            # self.participant.busy = False

    async def on_start_round(self, payload):
        payload = json.loads(payload)
        # print(message)
        await self.participant.start_round(payload)

    async def on_ask_success(self, payload):
        # message = json.loads(message)
        await self.participant.ask_success(payload)

    async def on_bid_success(self, payload):
        # message = json.loads(message)
        await self.participant.bid_success(payload)

    async def on_settled(self, payload):
        payload = json.loads(payload)
        await self.participant.settle_success(payload)

    async def on_return_extra_transactions(self, payload):
        payload = json.loads(payload)
        await self.participant.update_extra_transactions(payload)

    async def on_is_participant_joined(self, payload):
        await self.participant.is_participant_joined()

    async def on_start_episode(self, message):
        """Event triggers actions to be taken before the start of a simulation

        Args:
            message ([type]): [description]
        """
        
        self.participant.reset()
        if hasattr(self.participant, 'storage'):
            self.participant.storage.reset(soc_pct=0)
        # self.participant.trader.output_path = message['output_path']

        if hasattr(self.participant, 'records'):
            table_name = f'{str(message)}_{self.participant.market_id}'
            await self.participant.records.open_db(table_name)

    async def on_end_episode(self, message):
        # print("eog msg", message)
        message = json.loads(message)
        """Event triggers actions to be taken at the end of a simulation

        Args:
            message ([type]): [description]
        """
        if hasattr(self.participant, 'records'):
            # await asyncio.sleep(np.random.uniform(3, 30))
            await self.participant.records.ensure_records_complete()
            # self.participant.records.reset()

        # # TODO: save model
        # if hasattr(self.participant.trader, 'save_model'):
        #     await self.participant.trader.save_weights(**message)

        if hasattr(self.participant.trader, 'reset'):
            await self.participant.trader.reset(**message)

        # await self.participant.client.emit(event='participant_ready',
        #                                    data={self.participant.participant_id: True})
        self.participant.client.publish('/'.join([self.participant.market_id, 'simulation', 'participant_ready']),
                                        {self.participant.participant_id: True},
                                        user_property=('to', self.participant.market_sid))

    async def on_end_simulation(self):
        """Event tells the participant that it can terminate itself when ready.
        """
        # print('end_simulation')
        self.participant.run = False
        if hasattr(self.participant, 'records'):
            # await asyncio.sleep(np.random.uniform(3, 30))
            await self.participant.records.close_connection()
        await self.participant.kill()

    async def on_get_actions_return(self, payload):
        payload = json.loads(payload)
        # print(message)
        await self.participant.trader.get_actions_return(payload)

    async def on_get_metadata_return(self, payload):
        payload = json.loads(payload)
        # print(message)
        await self.participant.trader.get_metadata_return(payload)
