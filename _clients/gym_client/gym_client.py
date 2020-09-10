import asyncio
import socketio
from _utils import jkson
'''
'''
class Client:
    def __init__(self, server_address,configs):
        self.server_address = server_address
        self.sio_client = socketio.AsyncClient(
            reconnection=True,
            reconnection_attempts=100,
            reconnection_delay=1,
            reconnection_delay_max=5,
            randomization_factor=0.5,
            json=jkson
        )


    async def start_client(self):
        # this is the
        await self.sio_client.connect(self.server_address)
        await self.sio_client.wait()


    async def some_task(self):

    async def run(selfs):
    tasks = [
        # this is a list of asncio tasks
        asyncio.create_task() # inside this you can give it some functions to run
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
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--host', default=socket.gethostbyname(socket.getfqdn()), help='')
    parser.add_argument('--port', default=42069, help='')
    parser.add_argument('--config', default='', help='')
    args = parser.parse_args()
    client =Client(server_address=
                   configs )#TODO: Ask steven about this config file.
    loop = asyncio.get_event_loop()
    loop.run_until_complete(client.run())