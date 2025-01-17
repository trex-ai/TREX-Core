import asyncio
# from asyncio import Queue
import os
import json
import gmqtt
from gmqtt import Client as MQTTClient
from TREX_Core.participants.ns_common import NSDefault

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
        will_message = gmqtt.Message('/'.join([market_id, 'simulation', 'participant_disconnected']), participant_id,
                                     will_delay_interval=5)
        self.sio_client = MQTTClient(cuid(length=10).generate(), will_message=will_message)

        Participant = importlib.import_module('TREX_Core.participants.' + kwargs.get('type')).Participant
        self.participant = Participant(sio_client=self.sio_client,
                                       participant_id=participant_id,
                                       market_id=market_id,
                                       profile_db_path=profile_db_path,
                                       output_db_path=output_db_path,
                                       # trader_params=trader_params,
                                       # storage_params=storage_params,
                                       **kwargs)

        # self.msg_queue = Queue()
        self.ns = NSDefault(participant=self.participant)

    def on_connect(self, client, flags, rc, properties):
        market_id = self.participant.market_id
        participant_id = self.participant.participant_id
        loop = asyncio.get_running_loop()
        loop.create_task(self.ns.on_connect())
        # asyncio.run(keep_alive())
        print('Connected participant', market_id, participant_id)
        client.subscribe("/".join([market_id]), qos=0)
        client.subscribe("/".join([market_id, 'start_round']), qos=0)
        client.subscribe("/".join([market_id, participant_id]), qos=0)
        client.subscribe("/".join([market_id, participant_id, 'market_info']), qos=0)
        client.subscribe("/".join([market_id, participant_id, 'ask_ack']), qos=0)
        client.subscribe("/".join([market_id, participant_id, 'bid_ack']), qos=0)
        client.subscribe("/".join([market_id, participant_id, 'settled']), qos=0)
        client.subscribe("/".join([market_id, participant_id, 'extra_transaction']), qos=0)
        # client.subscribe("/".join([market_id, 'simulation', '+']), qos=0)
        client.subscribe("/".join([market_id, 'simulation', 'is_participant_joined']), qos=0)
        client.subscribe("/".join([market_id, 'simulation', 'start_episode']), qos=0)
        client.subscribe("/".join([market_id, 'simulation', 'end_episode']), qos=0)
        client.subscribe("/".join([market_id, 'simulation', 'end_simulation']), qos=0)

        client.subscribe("/".join([market_id, 'algorithm', participant_id, 'get_actions_return']), qos=2)
        # await keep_alive()

    # self.__client.publish('/'.join([self.market_id, 'simulation', 'participant_disconnected']), self.participant_id,
    #                       user_property=('to', self.market_sid))
    def on_disconnect(self, client, packet, exc=None):
        self.ns.on_disconnect()
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
        # await self.msg_queue.put(message)
        await self.ns.process_message(message)
        return 0
    # print(msg_queue)
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
            tg.create_task(self.run_client(self.sio_client))

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