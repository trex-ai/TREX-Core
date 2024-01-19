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

        self.interface = meshtastic.serial_interface.SerialInterface('COM5')

        self.count = 0
        # pub.subscribe(onReceive, "meshtastic.receive")

    def onReceive(packet, interface):  # called when a packet arrives
        global count
        # print(f"Received: {packet}")
        if 'decoded' in packet:
            decoded = packet['decoded']
            portnum = decoded['portnum']
            payload = decoded['payload']
            match portnum:
                case 'PRIVATE_APP':
                    count += 1
                    print(count)
                    # print(msgpack.loads(payload))

    async def process_msg_queue(self, queue):
        while True:
            message = await queue.get()
            await self.process_message(message)
            self.count += 1
            print(self.count)
            await asyncio.sleep(2)
    async def process_message(self, message):
        topic_event = message['topic'].split('/')[-1]
        payload = message['payload']
        # with self.interface as iface:
        self.interface.sendData(msgpack.dumps(payload))

    def on_connect(self, client, flags, rc, properties):
        market_id = 'training'
        participant_id = 'b1'
        print('Connected participant', market_id, participant_id)

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
        client.on_message = self.on_message
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


if __name__ == '__main__':

    client = Client(server_address='localhost')
    asyncio.run(client.run())
