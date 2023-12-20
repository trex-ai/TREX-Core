import asyncio
import socket
import socketio
from envController import EnvController


class NSDefault(socketio.AsyncClientNamespace):
    '''
    These namespaces allow you to set up the events and messages that are unique to the participant.
    They
    '''

    def __init__(self, envcontroller):
        super().__init__(namespace='')
        self.envController = envcontroller

    async def on_connect(self):
        await self.envController.register()


class NSSimulation(socketio.AsyncClientNamespace):
    def __init__(self, envcontroller):
        super().__init__(namespace='/simulation')

    async def on_connect(self):
        await self.envcontroller.register()

    async def on_end_generation(self, message):
        print('gym agent has ended generation')


class Client:
    def __init__(self, server_address):
        self.server_address = server_address
        self.sio_client = socketio.AsyncClient(
            reconnection=True,
            reconnection_attempts=3
        )
        self.envController = EnvController(self.sio_client)
        self.sio_client.register_namespace(NSDefault(self.envController))

    async def run(self):
        tasks = [
            asyncio.create_task(self.start_client())
        ]

        try:
            await asyncio.gather(*tasks)
        except SystemExit:
            for t in tasks:
                t.cancel()
            raise SystemExit

    async def start_client(self):
        await self.sio_client.connect(self.server_address)
        await self.sio_client.wait()


def __main():
    import socket
    import argparse
    host = socket.gethostbyname(socket.getfqdn())
    # print(host)
    hosts = ''.join(['http://', host, ':', str(42069)])
    print(hosts)
    client = Client(server_address=hosts)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(client.run())


if __name__ == "__main__":
    import sys

    sys.exit(__main())