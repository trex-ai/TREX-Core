import asyncio
import importlib
import json
import os

import socketio
import tenacity

from _clients.markets.ns_common import NSDefault
from _utils import jkson

if os.name == 'posix':
    import uvloop
    uvloop.install()

class Client:
    def __init__(self, server_address, market_configs):
        # Initialize client-server data
        self.server_address = server_address
        self.sio_client = socketio.AsyncClient(reconnection=True,
                                               reconnection_attempts=100,
                                               reconnection_delay=1,
                                               reconnection_delay_max=5,
                                               randomization_factor=0.5,
                                               json=jkson)

        market_configs = market_configs
        market_configs['market_id'] = market_configs.pop('id', '')
        grid_params = market_configs.pop('grid', {})

        # Initialize market information
        Market = importlib.import_module('_clients.markets.' + market_configs['type']).Market
        # NSMarket = importlib.import_module('_clients.markets.' + market_configs['type']).NSMarket

        self.market = Market(sio_client=self.sio_client,
                             **market_configs,
                             grid_params=grid_params)

        # register client in server rooms
        self.sio_client.register_namespace(NSDefault(self.market))
        # self.sio_client.register_namespace(NSMarket(self.market))
        # self.sio_client.register_namespace(NSSimulation(self.market))

        self.data_recorded = False
        self.recording_complete = False
    
    @tenacity.retry(wait=tenacity.wait_fixed(1))
    async def start_client(self):
        await self.sio_client.connect(self.server_address)
        await self.sio_client.wait()

    async def keep_alive(self):
        while True:
            await self.sio_client.sleep(10)
            await self.sio_client.emit("ping")

    async def run(self):
        tasks = [
            asyncio.create_task(self.start_client()),
            # asyncio.create_task(self.keep_alive()),
            asyncio.create_task(self.market.loop())]
        # try:
        await asyncio.gather(*tasks)
        # except SystemExit:
        #     for t in tasks:
        #         t.cancel()
        #     await self.sio_client.disconnect()
            # raise SystemExit

# def __main():
#     import socket
#     import argparse
#     parser = argparse.ArgumentParser(description='')
#     parser.add_argument('--host', default=socket.gethostbyname(socket.getfqdn()), help='')
#     parser.add_argument('--port', default=42069, help='')
#     parser.add_argument('--configs')
#     args = parser.parse_args()
#
#     client = Client(server_address=''.join(['http://', args.host, ':', str(args.port)]),
#                     market_configs=json.loads(args.configs))
#
#     loop = asyncio.get_event_loop()
#     loop.run_until_complete(client.run())

if __name__ == '__main__':
    import socket
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--host', default=socket.gethostbyname(socket.getfqdn()), help='')
    parser.add_argument('--port', default=42069, help='')
    parser.add_argument('--configs')
    args = parser.parse_args()

    client = Client(server_address=''.join(['http://', args.host, ':', str(args.port)]),
                    market_configs=json.loads(args.configs))

    loop = asyncio.get_event_loop()
    loop.run_until_complete(client.run())
    # import sys
    # sys.exit(__main())
