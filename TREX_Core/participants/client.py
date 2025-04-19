import asyncio
import os
import json
import gmqtt
from gmqtt import Client as MQTTClient

from cuid2 import Cuid as cuid

if os.name == 'posix':
    import uvloop
    uvloop.install()

STOP = asyncio.Event()

class Client:
    """A socket.io client wrapper for participants
    """
    def __init__(self, server_address, participant_id, market_id, profile_db_path, output_db_path, **kwargs):
        # Initialize client related data
        self.server_address = server_address
        # last will message
        # will_message = gmqtt.Message(f'{market_id}/simulation/participant_disconnected', participant_id,
        #                              will_delay_interval=5)
        will_message = gmqtt.Message(f'{market_id}/join_market/{participant_id}',
                                     '',
                                     retain=True,
                                     will_delay_interval=1)

        self.client = MQTTClient(cuid(length=10).generate(), will_message=will_message)
        Participant = importlib.import_module('TREX_Core.participants.' + kwargs.get('type')).Participant
        self.participant = Participant(client=self.client,
                                       participant_id=participant_id,
                                       market_id=market_id,
                                       profile_db_path=profile_db_path,
                                       output_db_path=output_db_path,
                                       # trader_params=trader_params,
                                       # storage_params=storage_params,
                                       **kwargs)

        self.msg_queue = asyncio.Queue()

        self.dispatch = {
            'market_info': self.on_update_market_info,
            'start_round': self.on_start_round,
            'ask_ack': self.on_ask_success,
            'bid_ack': self.on_bid_success,
            'settled': self.on_settled,
            'extra_transaction': self.on_return_extra_transactions,
            'is_participant_joined': self.on_is_participant_joined,
            'start_episode': self.on_start_episode,
            'end_episode': self.on_end_episode,
            'end_simulation': self.on_end_simulation,
            'get_actions_return': self.on_get_actions_return,
            'get_metadata_return': self.on_get_metadata_return
        }

    def on_connect(self, client, flags, rc, properties):
        market_id = self.participant.market_id
        participant_id = self.participant.participant_id
        asyncio.create_task(self.on_connect_task())

        # asyncio.run(keep_alive())
        print('Connected participant', market_id, participant_id)
        client.subscribe(f'{market_id}', qos=2)
        client.subscribe(f'{market_id}/start_round', qos=2)
        client.subscribe(f'{market_id}/{participant_id}', qos=2)
        client.subscribe(f'{market_id}/{participant_id}/market_info', qos=2)
        client.subscribe(f'{market_id}/{participant_id}/ask_ack', qos=2)
        client.subscribe(f'{market_id}/{participant_id}/bid_ack', qos=2)
        client.subscribe(f'{market_id}/{participant_id}/settled', qos=2)
        client.subscribe(f'{market_id}/{participant_id}/extra_transaction', qos=2)
        # client.subscribe("/".join([market_id, 'simulation', '+']), qos=0)
        client.subscribe(f'{market_id}/simulation/start_episode', qos=2)
        client.subscribe(f'{market_id}/simulation/is_participant_joined', qos=2)
        client.subscribe(f'{market_id}/simulation/end_episode', qos=2)
        client.subscribe(f'{market_id}/simulation/end_simulation', qos=2)

        client.subscribe(f'{market_id}/algorithm/{participant_id}/get_actions_return', qos=2)
        client.subscribe(f'{market_id}/algorithm/{participant_id}/get_metadata_return', qos=2)
        # await keep_alive()

    async def on_connect_task(self):
        # print('connected')
        await self.participant.open_profile_db()
        self.participant.server_online = True
        await self.participant.join_market()
    def on_disconnect(self, client, packet, exc=None):
        self.participant.server_online = False
        self.participant.busy = True
        print(self.participant.participant_id, 'disconnected')

    # def on_subscribe(self, client, mid, qos, properties):
    #     print('SUBSCRIBED')

    async def on_message(self, client, topic, payload, qos, properties):
        # print('participant RECV MSG:', topic, payload.decode(), properties)
        message = {
            'topic': topic,
            'payload': payload.decode(),
            'properties': properties
        }
        await self.msg_queue.put(message)
        # await self.ns.process_message(message)
        # return 0

    async def message_processor(self):
        while True:
            message = await self.msg_queue.get()
            try:
                await self.process_message(message)
            except Exception as e:
                logging.error(f"Error processing message: {e}", exc_info=True)
            finally:
                self.msg_queue.task_done()

    # print(msg_queue)
    async def process_message(self, message):
        for segment in reversed(message['topic'].split('/')):
            handler = self.dispatch.get(segment)
            if handler:
                await handler(message)
                break
        else:
            print("unrecognised topic:", message['topic'])

    async def on_update_market_info(self, message):
        client_data = json.loads(message['payload'])
        if client_data['id'] == self.participant.market_id:
            self.participant.market_sid = client_data['sid']
            self.participant.timezone = client_data['timezone']
            self.participant.market_connected = True
            # self.client.publish(f'{self.participant.market_id}/join_market/{self.participant.participant_id}',
            #                     '',
            #                     retain=True,
            #                     user_property=('to', '^all'))

    async def on_start_round(self, message):
        payload = json.loads(message['payload'])
        # print(message)
        await self.participant.start_round(payload)

    async def on_ask_success(self, message):
        # message = json.loads(message)
        await self.participant.ask_success(message['payload'])

    async def on_bid_success(self, message):
        # message = json.loads(message)
        await self.participant.bid_success(message['payload'])

    async def on_settled(self, message):
        payload = json.loads(message['payload'])
        await self.participant.settle_success(payload)

    async def on_return_extra_transactions(self, message):
        payload = json.loads(message['payload'])
        await self.participant.update_extra_transactions(payload)

    async def on_is_participant_joined(self, message):
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
            table_name = f'{message['payload']}_{self.participant.market_id}'
            await self.participant.records.open_db(table_name)

    async def on_end_episode(self, message):
        # print("eog msg", message)
        payload = json.loads(message['payload'])
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
            await self.participant.trader.reset(**payload)

        # await self.participant.client.emit(event='participant_ready',
        #                                    data={self.participant.participant_id: True})
        self.participant.client.publish(f'{self.participant.market_id}/simulation/participant_ready',
                                        {self.participant.participant_id: True},
                                        qos=2,
                                        user_property=('to', self.participant.market_sid))

    async def on_end_simulation(self, message):
        """Event tells the participant that it can terminate itself when ready.
        """
        # print('end_simulation')
        self.participant.run = False
        self.client.publish(f'{self.participant.market_id}/join_market/{self.participant.participant_id}',
                            '',
                            retain=True,
                            qos=2,
                            user_property=('to', '^all'))
        if hasattr(self.participant, 'records'):
            # await asyncio.sleep(np.random.uniform(3, 30))
            await self.participant.records.close_connection()
        await self.participant.kill()

    async def on_get_actions_return(self, message):
        payload = json.loads(message['payload'])
        # print(message)
        await self.participant.trader.get_actions_return(payload)

    async def on_get_metadata_return(self, message):
        payload = json.loads(message['payload'])
        # print(message)
        await self.participant.trader.get_metadata_return(payload)

    async def run_client(self, client):
        client.on_connect = self.on_connect
        client.on_disconnect = self.on_disconnect
        # client.on_subscribe = self.on_subscribe
        client.on_message = self.on_message

        # client.set_auth_credentials(token, None)
        # print(self.server_address)
        await client.connect(self.server_address, keepalive=60)
        await STOP.wait()

    async def run(self):
        """Function to start the client and other background tasks

        Raises:
            SystemExit: [description]
        """
        # tasks = [
        #     asyncio.create_task(self.run_client(self.sio_client))
        # ]
        # await asyncio.gather(*tasks)

        # for python 3.11+
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self.run_client(self.client))
            tg.create_task(self.message_processor())

    # def ask_exit(*args):
    #     STOP.set()

# async def main():

if __name__ == '__main__':
    # import sys
    # sys.exit(__main())
    import socket
    import argparse
    import importlib

    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('type', help='')
    parser.add_argument('--id', help='')
    parser.add_argument('--market_id', help='')
    parser.add_argument('--host', default="localhost", help='')
    parser.add_argument('--port', default=1883, help='')
    parser.add_argument('--profile_db_path', default=None, help='')
    parser.add_argument('--output_db_path', default=None, help='')
    # parser.add_argument('--trader', default=None, help='')
    # parser.add_argument('--storage', default=None, help='')
    # parser.add_argument('--generation_scale', default=1, help='')
    # parser.add_argument('--load_scale', default=1, help='')
    parser.add_argument('--configs')
    args = parser.parse_args()

    # server_address = ''.join(['http://', args.host, ':', str(args.port)])
    server_address = args.host
    client = Client(server_address=server_address,
                    # participant_type=args.type,
                    participant_id=args.id,
                    market_id=args.market_id,
                    profile_db_path=args.profile_db_path,
                    output_db_path=args.output_db_path,
                    # trader_params=args.trader,
                    # storage_params=args.storage,
                    # generation_scale=float(args.generation_scale),
                    # load_scale=float(args.load_scale),
                    **json.loads(args.configs)
                    )
    asyncio.run(client.run())

# parser.add_argument('--configs')
#     args = parser.parse_args()
#     # server_address = ''.join(['http://', args.host, ':', str(args.port)])
#     server_address = args.host
#     client = Client(server_address=server_address,
#                     market_configs=json.loads(args.configs))