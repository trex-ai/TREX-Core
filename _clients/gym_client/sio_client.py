import asyncio
import socketio
from _utils import jkson
import sys
from _clients.gym_client.ns_common import NSDefault, NSSimulation
from _clients.gym_client.gym_controller import Controller
sys.path.append("C:/source/TREX-Core")
'''
This is the code for the sio client that connects the gym to the simulation. 
'''
class Client:
    def __init__(self, server_address):
        self.server_address = server_address
        self.sio_client = socketio.AsyncClient(
            reconnection=True,
            reconnection_attempts=100,
            reconnection_delay=1,
            reconnection_delay_max=5,
            randomization_factor=0.5,
            json=jkson
        )
        gym_client= Controller(self.sio_client)
        self.sio_client.register_namespace(NSDefault(gym_client))
        self.sio_client.register_namespace(NSSimulation(gym_client))



    async def start_client(self):
        # this is the
        await self.sio_client.connect(self.server_address)
        await self.sio_client.wait()

    #example of additional tasks
    # async def some_task(self):
    #     some_crap = 0
    #     return True

    async def run(self):
        tasks = [
            # this is a list of asncio tasks
            asyncio.create_task(self.start_client()) # inside this you can give it some functions to run
            # the next task should mirror the
            # asyncio.create_task(self.gym.)
        ]

        try:
            await asyncio.gather(*tasks)
        except SystemExit:
            for t in tasks:
                t.cancel()
            raise SystemExit


def __main():
    import socket
    import argparse
    import json
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--host', default=socket.gethostbyname(socket.getfqdn()), help='')
    parser.add_argument('--port', default=42069, help='')
    # parser.add_argument('--config', default='', help='')
    args = parser.parse_args()
    # configs = json.loads(args.config)
    client = Client(server_address=''.join(['http://', args.host, ':', str(args.port)]))#This configs variable is the stuff that you put into the config under gym_client
    loop = asyncio.get_event_loop()
    loop.run_until_complete(client.run())

if __name__ == "__main__":
    import sys
    sys.exit(__main())