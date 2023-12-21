import asyncio
import os

import socketio
import tenacity
from TREX_Core._utils import jkson
from TREX_Core._clients.participants.ns_common import NSDefault

if os.name == 'posix':
    import uvloop
    uvloop.install()

class Client:
    """A socket.io client wrapper for participants
    """
    def __init__(self, server_address, participant_type, participant_id, market_id, db_path, trader_params, storage_params, **kwargs):
        # Initialize client related data
        self.server_address = server_address
        self.sio_client = socketio.AsyncClient(reconnection=True,
                                               reconnection_attempts=100,
                                               reconnection_delay=1,
                                               reconnection_delay_max=5,
                                               randomization_factor=0.5,
                                               json=jkson)

        Participant = importlib.import_module('TREX_Core._clients.participants.' + participant_type).Participant

        self.participant = Participant(sio_client=self.sio_client,
                                       participant_id=participant_id,
                                       market_id=market_id,
                                       db_path=db_path,
                                       trader_params=trader_params,
                                       storage_params=storage_params,
                                       # market_ns='_clients.participants.' + participant_type,
                                       **kwargs)


        self.sio_client.register_namespace(NSDefault(participant=self.participant))
        # self.sio_client.register_namespace(NSMarket(participant=self.participant))
        # self.sio_client.register_namespace(NSSimulation(participant=self.participant))
            
    # Continuously attempt to connect client
    @tenacity.retry(wait=tenacity.wait_fixed(1) + tenacity.wait_random(0, 2))
    async def start_client(self):
        """Function to connect client to server.
        """
        await self.sio_client.connect(self.server_address)
        await self.sio_client.wait()

    async def run(self):
        """Function to start the client and other background tasks

        Raises:
            SystemExit: [description]
        """

        tasks = [
            asyncio.create_task(self.start_client()),
            asyncio.create_task(self.participant.ping())]

        # try:
        await asyncio.gather(*tasks)
        # except SystemExit:
        #     for t in tasks:
        #         t.cancel()
        #     raise SystemExit

# def __main():
#     import socket
#     import argparse
#     import importlib
#
#     parser = argparse.ArgumentParser(description='')
#     parser.add_argument('type', help='')
#     parser.add_argument('--id', help='')
#     parser.add_argument('--market_id', help='')
#     parser.add_argument('--host', default=socket.gethostbyname(socket.getfqdn()), help='')
#     parser.add_argument('--port', default=42069, help='')
#     parser.add_argument('--db_path', default=None, help='')
#     parser.add_argument('--trader', default=None, help='')
#     parser.add_argument('--storage', default=None, help='')
#     parser.add_argument('--generation_scale', default=1, help='')
#     parser.add_argument('--load_scale', default=1, help='')
#     args = parser.parse_args()
#
#     client = Client(''.join(['http://', args.host, ':', str(args.port)]),
#                     participant_type=args.type,
#                     participant_id=args.id,
#                     market_id=args.market_id,
#                     db_path=args.db_path,
#                     trader_params=args.trader,
#                     storage_params=args.storage,
#                     generation_scale=float(args.generation_scale),
#                     load_scale=float(args.load_scale),
#                     )
#
#     loop = asyncio.get_running_loop()
#     loop.run_until_complete(client.run())

if __name__ == '__main__':
    # import sys
    # sys.exit(__main())
    import socket
    import argparse
    import importlib

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('type', help='')
    parser.add_argument('--id', help='')
    parser.add_argument('--market_id', help='')
    parser.add_argument('--host', default="localhost", help='')
    parser.add_argument('--port', default=42069, help='')
    parser.add_argument('--db_path', default=None, help='')
    parser.add_argument('--trader', default=None, help='')
    parser.add_argument('--storage', default=None, help='')
    parser.add_argument('--generation_scale', default=1, help='')
    parser.add_argument('--load_scale', default=1, help='')
    args = parser.parse_args()

    client = Client(''.join(['http://', args.host, ':', str(args.port)]),
                    participant_type=args.type,
                    participant_id=args.id,
                    market_id=args.market_id,
                    db_path=args.db_path,
                    trader_params=args.trader,
                    storage_params=args.storage,
                    generation_scale=float(args.generation_scale),
                    load_scale=float(args.load_scale),
                    )

    asyncio.run(client.run())
