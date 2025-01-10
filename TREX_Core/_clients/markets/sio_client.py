import asyncio
import json
# from asyncio import Queue
import os
from cuid2 import Cuid as cuid
from gmqtt import Client as MQTTClient

from TREX_Core._clients.markets.ns_common import NSDefault

if os.name == 'posix':
    import uvloop
    uvloop.install()

STOP = asyncio.Event()

class Client:
    def __init__(self, server_address, market_configs):
        # Initialize client-server data
        self.server_address = server_address
        market_configs = market_configs
        market_configs['market_id'] = market_configs.pop('id', '')
        grid_params = market_configs.pop('grid', {})

        self.sio_client = MQTTClient(cuid(length=10).generate())

        # Initialize market information
        try:
            Market = importlib.import_module('markets.' + market_configs['type']).Market
        except ImportError:
            Market = importlib.import_module('TREX_Core._clients.markets.' + market_configs['type']).Market


        self.market = Market(sio_client=self.sio_client,
                             **market_configs,
                             grid_params=grid_params)

        # self.msg_queue = Queue()
        self.ns = NSDefault(self.market)

        self.data_recorded = False
        self.recording_complete = False

    def on_connect(self, client, flags, rc, properties):
        market_id = self.market.market_id
        print('Connected market', market_id)
        client.subscribe("/".join([market_id]), qos=0)
        # client.subscribe("/".join([market_id, '+']), qos=0)
        client.subscribe("/".join([market_id, market_id]), qos=0)
        client.subscribe("/".join([market_id, 'join_market']), qos=0)
        client.subscribe("/".join([market_id, 'bid']), qos=0)
        client.subscribe("/".join([market_id, 'ask']), qos=0)
        client.subscribe("/".join([market_id, 'settlement_delivered']), qos=0)
        client.subscribe("/".join([market_id, 'meter']), qos=0)

        # client.subscribe("/".join([market_id, 'simulation', '+']), qos=0)
        client.subscribe("/".join([market_id, 'simulation', 'start_round']), qos=0)
        client.subscribe("/".join([market_id, 'simulation', 'start_generation']), qos=0)
        client.subscribe("/".join([market_id, 'simulation', 'end_generation']), qos=0)
        client.subscribe("/".join([market_id, 'simulation', 'end_simulation']), qos=0)
        client.subscribe("/".join([market_id, 'simulation', 'is_market_online']), qos=0)
        # participant_id = self.participant.participant_id
        # loop = asyncio.get_running_loop()
        # loop.create_task(self.ns.on_connect())

    def on_disconnect(self, client, packet, exc=None):
        # self.ns.on_disconnect()
        print('market disconnected')

    # def on_subscribe(self, client, mid, qos, properties):
    #     print('SUBSCRIBED')

    async def on_message(self, client, topic, payload, qos, properties):
        # print('market RECV MSG:', topic, payload.decode(), properties)
        message = {
            'topic': topic,
            'payload': payload.decode(),
            'properties': properties
        }

        # await self.msg_queue.put(msg)
        await self.ns.process_message(message)
        return 0
    # @tenacity.retry(wait=tenacity.wait_fixed(1))
    # async def start_client(self):
    #     await self.sio_client.connect(self.server_address)
    #     await self.sio_client.wait()


    # async def keep_alive(self):
    #     while self.market.server_online:
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
        # await asyncio.wait()
        # await client.disconnect()

    # async def run(self):
    #     tasks = [
    #         asyncio.create_task(self.start_client()),
    #         asyncio.create_task(self.keep_alive()),
    #         asyncio.create_task(self.market.loop())]
    #     # try:
    #     await asyncio.gather(*tasks)
    #     # except SystemExit:
    #     #     for t in tasks:
    #     #         t.cancel()
    #     #     await self.sio_client.disconnect()
    #         # raise SystemExit

    async def run(self):
        """Function to start the client and other background tasks

        Raises:
            SystemExit: [description]
        """
        # tasks = [
        #     # asyncio.create_task(keep_alive()),
        #     # asyncio.create_task(self.ns.listen(self.msg_queue)),
        #     asyncio.create_task(self.run_client(self.sio_client)),
        #     asyncio.create_task(self.market.loop())
        # ]
        #
        # # try:
        # await asyncio.gather(*tasks)

        # for python 3.11+
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self.run_client(self.sio_client))
            # tg.create_task(self.market.loop())

if __name__ == '__main__':
    import argparse
    import importlib

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--host', default="localhost", help='')
    parser.add_argument('--port', default=1883, help='')
    parser.add_argument('--configs')
    args = parser.parse_args()
    # server_address = ''.join(['http://', args.host, ':', str(args.port)])
    server_address = args.host
    client = Client(server_address=server_address,
                    market_configs=json.loads(args.configs))

    asyncio.run(client.run())
