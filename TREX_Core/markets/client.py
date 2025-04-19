import asyncio
import json
# from asyncio import Queue
import os
import signal
from cuid2 import Cuid as cuid
from gmqtt import Client as MQTTClient

# if os.name == 'posix':
#     import uvloop
#
#     uvloop.install()

STOP = asyncio.Event()


class Client:
    def __init__(self, server_address, market_configs):
        # Initialize client-server data
        self.server_address = server_address
        market_configs = market_configs
        market_configs['market_id'] = market_configs.pop('id', '')
        grid_params = market_configs.pop('grid', {})

        self.client = MQTTClient(cuid(length=10).generate())

        # Initialize market information
        try:
            Market = importlib.import_module('markets.' + market_configs['type']).Market
        except ImportError:
            Market = importlib.import_module('TREX_Core.markets.' + market_configs['type']).Market

        self.market = Market(client=self.client,
                             **market_configs,
                             grid_params=grid_params)

        self.data_recorded = False
        self.recording_complete = False
        self.msg_queue = asyncio.Queue()

        self.dispatch = {
            "join_market":          self.on_participant_connected,
            "bid":                  self.on_bid,
            "ask":                  self.on_ask,
            "settlement_delivered": self.on_settlement_delivered,
            "meter":                self.on_meter_data,
            "start_round":          self.on_start_round,
            "start_episode":        self.on_start_episode,
            "end_episode":          self.on_end_episode,
            "end_simulation":       self.on_end_simulation,
            "is_market_online":     self.on_is_market_online,
        }

    def on_connect(self, client, flags, rc, properties):
        market_id = self.market.market_id
        print('Connected market', market_id)
        client.subscribe(f'{market_id}', qos=2)
        # client.subscribe(f'{market_id}/+', qos=2)
        client.subscribe(f'{market_id}/join_market/+', qos=2)
        # client.subscribe(f'{market_id}/simulation/+', qos=2)

        client.subscribe(f'{market_id}/{market_id}', qos=2)
        client.subscribe(f'{market_id}/bid', qos=2)
        client.subscribe(f'{market_id}/ask', qos=2)
        client.subscribe(f'{market_id}/settlement_delivered', qos=2)
        client.subscribe(f'{market_id}/meter', qos=2)

        client.subscribe(f'{market_id}/simulation/start_round', qos=2)
        client.subscribe(f'{market_id}/simulation/start_episode', qos=2)
        client.subscribe(f'{market_id}/simulation/end_episode', qos=2)
        client.subscribe(f'{market_id}/simulation/end_simulation', qos=2)
        client.subscribe(f'{market_id}/simulation/is_market_online', qos=2)

    def on_disconnect(self, client, packet, exc=None):
        # self.market.server_online = False
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

        await self.msg_queue.put(message)
        # print('received', message)
        # await self.process_message(message)
        # return 0

    async def message_processor(self):
        while True:
            message = await self.msg_queue.get()
            # print('processed', message)
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

    async def on_bid(self, message):
        bid = json.loads(message['payload'])
        try:
            entry_id, participant_id, participant_sid = await self.market.submit_bid(bid)
            self.client.publish(f'{self.market.market_id}/{participant_id}/bid_ack', entry_id,
                                user_property=('to', participant_sid),
                                qos=2)
        except TypeError:
            return

    async def on_ask(self, message):
        ask = json.loads(message['payload'])
        try:
            entry_id, participant_id, participant_sid = await self.market.submit_ask(ask)
            self.client.publish(f'{self.market.market_id}/{participant_id}/ask_ack', entry_id,
                                user_property=('to', participant_sid),
                                qos=2)
        except TypeError:
            return

    async def on_settlement_delivered(self, message):
        payload = json.loads(message['payload'])
        await self.market.settlement_delivered(payload)

    async def on_meter_data(self, message):
        # print("meter data")
        payload = json.loads(message['payload'])
        await self.market.meter_data(payload)

    async def on_participant_connected(self, message):
        # print(type(client_data))
        if message['payload']:
            client_data = json.loads(message['payload'])
            client_data['id'] = message['topic'].split('/')[-1]
            market_id, market_sid, timezone = await self.market.participant_connected(client_data)
            # async def participant_connected(self, client_data):

            self.client.publish(f'{self.market.market_id}/{client_data['id']}/market_info',
                                {'id': market_id,
                                 'sid': market_sid,
                                 'timezone': timezone, },
                                user_property=('to', client_data['sid']),
                                qos=2)

    async def on_is_market_online(self, message):
        self.client.publish(f'{self.market.market_id}/simulation/market_online', self.market.market_id, qos=2)
        # await self.market.market_is_online()

    async def on_start_round(self, message):
        payload = json.loads(message['payload'])
        # await self.market.step(message['duration'], sim_params=message)
        step_task = asyncio.create_task(self.market.step(payload['duration'], sim_params=payload))
        step_task.add_done_callback(self.on_round_done)
        # self.client.publish(f'{self.market.market_id}/simulation/end_round', self.market.market_id, qos=0)

    def on_round_done(self, task):
        self.client.publish(f'{self.market.market_id}/simulation/end_round', self.market.market_id, qos=2)

    async def on_start_episode(self, message):
        # message = json.loads(message)
        table_name = f'{message['payload']}_{self.market.market_id}'
        await self.market.open_db(table_name)

    async def on_end_episode(self, message):
        # await self.market.end_sim_generation()
        # No need to call record_transactions here since ensure_transactions_complete already does it
        await self.market.ensure_transactions_complete()
        # if not last_generation:
        await self.market.reset_market()
        self.client.publish(f'{self.market.market_id}/simulation/market_ready', self.market.market_id, qos=2)

    async def on_end_simulation(self, message):
        # print('end simulation')
        self.market.run = False
        # await self.market.end_sim_generation()
        # No need to call record_transactions here since ensure_transactions_complete already does it
        await self.market.ensure_transactions_complete()
        # Close the database connection after transactions are complete but before disconnecting
        await self.market.close_connection()
        # No need for sleep - all database tasks are confirmed complete
        await self.client.disconnect()
        # print('attempting to end')
        os.kill(os.getpid(), signal.SIGINT)
        raise SystemExit

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
        # for python 3.11+
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self.run_client(self.client))
            for _ in range(4):
                tg.create_task(self.message_processor())  # N consumers
            # tg.create_task(self.message_processor())


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
