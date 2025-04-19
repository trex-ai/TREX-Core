import asyncio
import os
import json
from pprint import pprint

from gmqtt import Client as MQTTClient
from TREX_Core.sim_controller.sim_controller import Controller
from cuid2 import Cuid as cuid

if os.name == 'posix':
    import uvloop
    uvloop.install()

STOP = asyncio.Event()

class Client:
    # Initialize client data for sim controller
    def __init__(self, server_address, config):
        self.server_address = server_address
        self.client = MQTTClient(cuid(length=10).generate())

        # Set client to controller class
        self.controller = Controller(self.client, config)
        self.msg_queue = asyncio.Queue()
        # self._write_state_lock = asyncio.Lock()

        self.dispatch = {
            'market_online': self.on_market_online,
            'participant_joined': self.on_participant_joined,
            'end_turn': self.on_end_turn,
            'end_round': self.on_end_round,
            'participant_ready': self.on_participant_ready,
            'market_ready': self.on_market_ready,
            'participant_disconnected': self.on_participant_disconnected,
            'policy_server_ready': self.on_policy_server_ready,
            'sim_controller_status': self.on_sim_controller_status,
        }

    def on_connect(self, client, flags, rc, properties):
        market_id = self.controller.market_id
        print('Connected sim_controller', market_id)
        asyncio.create_task(self.on_connect_task())

        # client.subscribe("/".join([market_id]), qos=0)
        # client.subscribe("/".join([market_id, 'simulation', '+']), qos=0)
        client.subscribe(f'{market_id}/simulation/market_online', qos=2)
        client.subscribe(f'{market_id}/simulation/participant_joined', qos=2)
        client.subscribe(f'{market_id}/simulation/end_turn', qos=2)
        client.subscribe(f'{market_id}/simulation/end_round', qos=2)
        client.subscribe(f'{market_id}/simulation/participant_ready', qos=2)
        client.subscribe(f'{market_id}/simulation/market_ready', qos=2)
        client.subscribe(f'{market_id}/algorithm/policy_server_ready', qos=2)
        client.subscribe(f'debug/sim_controller_status', qos=2)

    async def on_connect_task(self):
        await self.controller.register()

    def on_disconnect(self, client, packet, exc=None):
        # self.ns.on_disconnect()
        print('sim controller disconnected')

    async def on_message(self, client, topic, payload, qos, properties):
        # print('controller RECV MSG:', topic, payload.decode(), properties)
        message = {
            'topic': topic,
            'payload': payload.decode(),
            'properties': properties
        }
        await self.msg_queue.put(message)
        # await self.ns.process_message(message)
        # return 0

    async def message_processor(self):
        while True:
            message = await self.msg_queue.get()
            try:
                await self.process_message(message)
            except Exception as e:
                logging.error(f"Error processing message: {e}", exc_info=True)
            finally:
                self.msg_queue.task_done()

    async def process_message(self, message):
        for segment in reversed(message['topic'].split('/')):
            handler = self.dispatch.get(segment)
            if handler:
                await handler(message)
                break
        else:
            print("unrecognised topic:", message['topic'])

    async def on_participant_joined(self, message):
        participant_id = message['payload']
        # async with self._write_state_lock:
        await self.controller.participant_online(participant_id, True)

    async def on_participant_disconnected(self, message):
        print(message['payload'], 'PARTICIPANT LOST')
        participant_id = message['payload']
        # async with self._write_state_lock:
        await self.controller.participant_online(participant_id, False)

    async def on_participant_ready(self, message):
        payload = json.loads(message['payload'])
        print(payload)
        for participant_id in payload:
            await self.controller.participant_status(participant_id, 'ready', payload[participant_id])

    async def on_participant_weights_loaded(self, message):
        payload = message['payload']
        for participant_id in payload:
            await self.controller.participant_status(participant_id, 'weights_loaded', payload[participant_id])

    # send by individual participants
    async def on_end_turn(self, message):
        # async with self._write_state_lock:
        task = asyncio.create_task(self.controller.update_turn_status(message['payload']))

    # sent by the market
    async def on_end_round(self, message):
        await self.controller.market_turn_end()
        # async with self._write_state_lock:
        task = asyncio.create_task(self.controller.update_turn_status(message['payload']))

    async def on_market_online(self, message):
        self.controller.status['market_online'] = True

    async def on_market_ready(self, message):
        self.controller.status['market_ready'] = True

    async def on_policy_server_ready(self, message):
        self.controller.status['policy_server_ready'] = True
        await self.controller.update_turn_status(message['payload'])

    async def on_sim_controller_status(self, message):
        pprint(self.controller.status)
    # print(msg_queue)
    async def run_client(self, client):
        client.on_connect = self.on_connect
        client.on_disconnect = self.on_disconnect
        # client.on_subscribe = self.on_subscribe
        client.on_message = self.on_message

        # client.set_auth_credentials(token, None)
        # print(self.server_address)
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
            tg.create_task(self.run_client(self.client))
            tg.create_task(self.controller.monitor())
            # tg.create_task(self.message_processor())
            for _ in range(4):
                tg.create_task(self.message_processor())

    # except SystemExit:
    #     for t in tasks:
    #         t.cancel()
    #     raise SystemExit

if __name__ == '__main__':
    # import sys
    # sys.exit(__main())
    import socket
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--host', default="localhost", help='')
    parser.add_argument('--port', default=1883, help='')
    parser.add_argument('--config', default='', help='')
    args = parser.parse_args()

    server_address = args.host
    client = Client(server_address=server_address,
                    config=json.loads(args.config))

    asyncio.run(client.run())
