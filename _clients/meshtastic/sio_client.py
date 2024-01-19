import asyncio
from asyncio import Queue
import os

from gmqtt import Client as MQTTClient
# from _clients.participants.ns_common import NSDefault

from cuid2 import Cuid as cuid

import meshtastic.serial_interface
from pubsub import pub
import msgpack

if os.name == 'posix':
    import uvloop
    uvloop.install()

STOP = asyncio.Event()

class Client:
    """A socket.io client wrapper for participants
    """
    def __init__(self, server_address, **kwargs):
        # Initialize client related data
        self.server_address = server_address
        self.sio_client = MQTTClient(cuid(length=10).generate())

        # Participant = importlib.import_module('_clients.participants.' + participant_type).Participant
        # self.participant = Participant(sio_client=self.sio_client,
        #                                participant_id=participant_id,
        #                                market_id=market_id,
        #                                db_path=db_path,
        #                                trader_params=trader_params,
        #                                storage_params=storage_params,
        #                                **kwargs)

        self.msg_queue = Queue()
        # self.ns = NSDefault()

        self.interface = meshtastic.serial_interface.SerialInterface('COM6')

        self.count = 0
        self.ack = True
        # pub.subscribe(onReceive, "meshtastic.receive")

    async def process_msg_queue(self, queue):
        while True:
            message = await queue.get()
            if self.ack:
                self.ack = False
                self.count += 1
                await self.process_message(message)
                # self.count += 1
                print(self.count, self.msg_queue.qsize())
                # if not self.count % 50:
                #     await asyncio.sleep(10)
            # else:
            await asyncio.sleep(0.5)
    def on_ack(self, packet):
        # print(packet)
        self.ack = True
    async def process_message(self, message):
        # topic_event = message['topic'].split('/')[-1]
        payload = message['payload']
        if type(payload) is dict:
            payload['count'] = self.count
        else:
            payload = {'count': self.count}
        # with self.interface as iface:
        self.interface.sendData(msgpack.dumps(payload), portNum=68, destinationId='!3423c0ee', wantAck=True,
                                onResponse=self.on_ack)
        # self.interface.sendData(msgpack.dumps(payload), portNum=68, destinationId='!3423c0ee', wantAck=False)
        # self.interface.se

    def on_connect(self, client, flags, rc, properties):
        market_id = 'training'
        participant_id = 's1'
        # loop = asyncio.get_running_loop()
        # loop.create_task(self.ns.on_connect())
        # asyncio.run(keep_alive())
        print('Connected participant', market_id, participant_id)
        # client.subscribe("/".join([market_id]), qos=0)
        client.subscribe("/".join([market_id, 'join_market']), qos=0)
        client.subscribe("/".join([market_id, 'bid']), qos=0)
        client.subscribe("/".join([market_id, 'ask']), qos=0)
        client.subscribe("/".join([market_id, 'settlement_delivered']), qos=0)
        client.subscribe("/".join([market_id, 'meter_data']), qos=0)
        client.subscribe("/".join([market_id, 'simulation', 'end_turn']), qos=0)
        client.subscribe("/".join([market_id, 'simulation', 'participant_joined']), qos=0)

        # self.interface.sendText("hello mesh")
        #
        # client.subscribe("/".join([market_id, participant_id]), qos=0)
        # client.subscribe("/".join([market_id, participant_id, 'update_market_info']), qos=0)
        # client.subscribe("/".join([market_id, participant_id, 'ask_success']), qos=0)
        # client.subscribe("/".join([market_id, participant_id, 'bid_success']), qos=0)
        # client.subscribe("/".join([market_id, participant_id, 'settled']), qos=0)
        # client.subscribe("/".join([market_id, participant_id, 'return_extra_transaction']), qos=0)
        # # client.subscribe("/".join([market_id, 'simulation', '+']), qos=0)
        # client.subscribe("/".join([market_id, 'simulation', 'is_participant_joined']), qos=0)
        # client.subscribe("/".join([market_id, 'simulation', 'start_generation']), qos=0)
        # client.subscribe("/".join([market_id, 'simulation', 'end_generation']), qos=0)
        # client.subscribe("/".join([market_id, 'simulation', 'end_simulation']), qos=0)
        # await keep_alive()
    # def on_disconnect(self, client, packet, exc=None):
    #     self.ns.on_disconnect()
    #     print(self.participant.participant_id, 'disconnected')

    def on_subscribe(self, client, mid, qos, properties):
        print('SUBSCRIBED')

    async def on_message(self, client, topic, payload, qos, properties):
        # print('participant RECV MSG:', topic, payload.decode(), properties)
        # print(payload)
        message = {
            'topic': topic,
            'payload': payload.decode(),
            'properties': properties
        }
        await self.msg_queue.put(message)
        # loop = asyncio.get_running_loop()
        # await loop.create_task(self.process_message(message))
        # await self.process_message(message)
    # print(msg_queue)
    async def run_client(self, client):
        client.on_connect = self.on_connect
        # client.on_disconnect = self.on_disconnect
        # client.on_subscribe = self.on_subscribe
        client.on_message = self.on_message

        # client.set_auth_credentials(token, None)
        # print(self.server_address)
        await client.connect(self.server_address)
        await STOP.wait()

    async def run(self):
        """Function to start the client and other background tasks

        Raises:
            SystemExit: [description]
        """
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self.run_client(self.sio_client))
            tg.create_task(self.process_msg_queue(self.msg_queue))

    # def ask_exit(*args):
    #     STOP.set()

# async def main():

if __name__ == '__main__':
    # import sys
    # sys.exit(__main())
    import socket
    import argparse
    import importlib

    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('type', help='')
    # parser.add_argument('--id', help='')
    # parser.add_argument('--market_id', help='')
    # parser.add_argument('--host', default=socket.gethostbyname(socket.getfqdn()), help='')
    # parser.add_argument('--port', default=42069, help='')
    # parser.add_argument('--db_path', default=None, help='')
    # parser.add_argument('--trader', default=None, help='')
    # parser.add_argument('--storage', default=None, help='')
    # parser.add_argument('--generation_scale', default=1, help='')
    # parser.add_argument('--load_scale', default=1, help='')
    # args = parser.parse_args()

    # server_address = ''.join(['http://', args.host, ':', str(args.port)])
    # server_address = args.host
    client = Client(server_address='localhost')
    asyncio.run(client.run())
