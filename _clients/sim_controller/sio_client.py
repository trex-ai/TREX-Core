import asyncio
import os

import socketio
import tenacity
from TREX_Core._utils import jkson
from TREX_Core._clients.sim_controller.sim_controller import Controller
from TREX_Core._clients.sim_controller.ns_common import NSDefault
from TREX_Core._clients.sim_controller.sim_controller import NSMarket, NSSimulation

# from _clients.sim_controller.sim_controller import NSMarket, NSSimulation

if os.name == 'posix':
    import uvloop
    uvloop.install()

class Client:
    # Initialize client data for sim controller
    def __init__(self, server_address, configs):
        self.server_address = server_address
        self.sio_client = socketio.AsyncClient(reconnection=True,
                                               reconnection_attempts=100,
                                               reconnection_delay=1,
                                               reconnection_delay_max=30,
                                               randomization_factor=0.5,
                                               json=jkson)

        # Set client to controller class
        self.controller = Controller(self.sio_client, configs)
        self.sio_client.register_namespace(NSDefault(controller=self.controller))
        # self.sio_client.register_namespace(NSMarket(controller=self.controller))
        # self.sio_client.register_namespace(NSSimulation(controller=self.controller))

    @tenacity.retry(wait=tenacity.wait_fixed(1) + tenacity.wait_random(0, 2))
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
            asyncio.create_task(self.controller.monitor())
        ]

        # try:
        await asyncio.gather(*tasks)
        # except SystemExit:
        #     for t in tasks:
        #         t.cancel()
        #     raise SystemExit

# def __main():
#     import socket
#     import argparse
#     import json
#     parser = argparse.ArgumentParser(description='')
#     parser.add_argument('--host', default=socket.gethostbyname(socket.getfqdn()), help='')
#     parser.add_argument('--port', default=42069, help='')
#     parser.add_argument('--config', default='', help='')
#     args = parser.parse_args()
#
#     configs = json.loads(args.config)
#     client = Client(server_address=''.join(['http://', args.host, ':', str(args.port)]),
#                     configs=configs)
#
#     loop = asyncio.get_running_loop()
#     loop.run_until_complete(client.run())

if __name__ == '__main__':
    # import sys
    # sys.exit(__main())
    import socket
    import argparse
    import json

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--host', default=socket.gethostbyname(socket.getfqdn()), help='')
    parser.add_argument('--port', default=42069, help='')
    parser.add_argument('--config', default='', help='')
    args = parser.parse_args()

    configs = json.loads(args.config)
    client = Client(server_address=''.join(['http://', args.host, ':', str(args.port)]),
                    configs=configs)

    asyncio.run(client.run())
