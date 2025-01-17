import asyncio
# from asyncio import Queue
import os
import json

from gmqtt import Client as MQTTClient
from TREX_Core.sim_controller.ns_common import NSDefault
from TREX_Core.sim_controller.sim_controller import Controller
# from _utils import jkson
# from _clients.sim_controller.sim_controller import NSMarket, NSSimulation
from cuid2 import Cuid as cuid

if os.name == 'posix':
    import uvloop
    uvloop.install()

STOP = asyncio.Event()

class Client:
    # Initialize client data for sim controller
    def __init__(self, server_address, config):
        self.server_address = server_address
        self.sio_client = MQTTClient(cuid(length=10).generate())

        # Set client to controller class
        self.controller = Controller(self.sio_client, config)
        # self.msg_queue = Queue()
        self.ns = NSDefault(self.controller)

    def on_connect(self, client, flags, rc, properties):
        market_id = self.controller.market_id
        print('Connected sim_controller', market_id)
        loop = asyncio.get_running_loop()
        loop.create_task(self.ns.on_connect())

        # client.subscribe("/".join([market_id]), qos=0)
        # client.subscribe("/".join([market_id, 'simulation', '+']), qos=0)
        client.subscribe("/".join([market_id, 'simulation', 'market_online']), qos=0)
        client.subscribe("/".join([market_id, 'simulation', 'participant_joined']), qos=0)
        client.subscribe("/".join([market_id, 'simulation', 'end_turn']), qos=0)
        client.subscribe("/".join([market_id, 'simulation', 'end_round']), qos=0)
        client.subscribe("/".join([market_id, 'simulation', 'participant_ready']), qos=0)
        client.subscribe("/".join([market_id, 'simulation', 'market_ready']), qos=0)
        client.subscribe("/".join([market_id, 'algorithm', 'policy_sever_ready']), qos=2)
        client.subscribe("/".join(['debug', 'sim_controller_status']), qos=0)

    def on_disconnect(self, client, packet, exc=None):
        # self.ns.on_disconnect()
        print('sim controller disconnected')

    # def on_subscribe(self, client, mid, qos, properties):
    #     print('SUBSCRIBED')

    async def on_message(self, client, topic, payload, qos, properties):
        # print('controller RECV MSG:', topic, payload.decode(), properties)
        message = {
            'topic': topic,
            'payload': payload.decode(),
            'properties': properties
        }
        # await self.msg_queue.put(message)
        await self.ns.process_message(message)
        return 0

    # @tenacity.retry(wait=tenacity.wait_fixed(1) + tenacity.wait_random(0, 2))
    # async def start_client(self):
    #     await self.sio_client.connect(self.server_address)
    #     await self.sio_client.wait()
    #
    # async def keep_alive(self):
    #     while True:
    #         await self.sio_client.sleep(10)
    #         await self.sio_client.emit("ping")
    async def run_client(self, client):
        client.on_connect = self.on_connect
        client.on_disconnect = self.on_disconnect
        # client.on_subscribe = self.on_subscribe
        client.on_message = self.on_message

        # client.set_auth_credentials(token, None)
        await client.connect(self.server_address, keepalive=60)
        await STOP.wait()


    async def run(self):
        """Function to start the client and other background tasks

        Raises:
            SystemExit: [description]
        """
        # tasks = [
        #     # asyncio.create_task(keep_alive()),
        #     # asyncio.create_task(self.ns.listen(self.msg_queue)),
        #     asyncio.create_task(self.run_client(self.sio_client)),
        #     asyncio.create_task(self.controller.monitor())
        # ]
        #
        # # try:
        # await asyncio.gather(*tasks)

        async with asyncio.TaskGroup() as tg:
            tg.create_task(self.run_client(self.sio_client))
            tg.create_task(self.controller.monitor())
    # except SystemExit:
    #     for t in tasks:
    #         t.cancel()
    #     raise SystemExit

if __name__ == '__main__':
    # import sys
    # sys.exit(__main())
    import socket
    import argparse
    import json

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--host', default="localhost", help='')
    parser.add_argument('--port', default=1883, help='')
    parser.add_argument('--config', default='', help='')
    args = parser.parse_args()

    server_address = args.host
    client = Client(server_address=server_address,
                    config=json.loads(args.config))

    asyncio.run(client.run())
