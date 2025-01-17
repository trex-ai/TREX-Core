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
        client.subscribe("/".join([market_id, 'simulation', 'start_episode']), qos=0)
        client.subscribe("/".join([market_id, 'simulation', 'end_episode']), qos=0)
        client.subscribe("/".join([market_id, 'simulation', 'end_simulation']), qos=0)
        client.subscribe("/".join([market_id, 'simulation', 'is_market_online']), qos=0)

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

        # await self.msg_queue.put(msg)
        await self.process_message(message)
        return 0

    async def process_message(self, message):
        # if self.market.run:
        topic_event = message['topic'].split('/')[-1]
        payload = message['payload']

        match topic_event:
            # market related events
            case 'join_market':
                await self.on_participant_connected(payload)
            case 'bid':
                await self.on_bid(payload)
            case 'ask':
                await self.on_ask(payload)
            case 'settlement_delivered':
                await self.on_settlement_delivered(payload)
            case 'meter':
                # print("METER DATA")
                await self.on_meter_data(payload)
            # simulation related events
            case 'start_round':
                await self.on_start_round(payload)
            case 'start_episode':
                await self.on_start_generation(payload)
            case 'end_episode':
                await self.on_end_generation(payload)
            case 'end_simulation':
                await self.on_end_simulation()
            case 'is_market_online':
                await self.on_is_market_online()

    async def on_bid(self, bid):
        bid = json.loads(bid)
        entry_id, participant_id, participant_sid = await self.market.submit_bid(bid)
        self.client.publish('/'.join([self.market.market_id, participant_id, 'bid_ack']), entry_id,
                            user_property=('to', participant_sid))

    async def on_ask(self, ask):
        ask = json.loads(ask)
        entry_id, participant_id, participant_sid = await self.market.submit_ask(ask)
        self.client.publish('/'.join([self.market.market_id, participant_id, 'ask_ack']), entry_id,
                            user_property=('to', participant_sid))

    async def on_settlement_delivered(self, message):
        message = json.loads(message)
        await self.market.settlement_delivered(message)

    async def on_meter_data(self, message):
        # print("meter data")
        message = json.loads(message)
        await self.market.meter_data(message)

    async def on_participant_connected(self, message):
        # print(type(client_data))
        client_data = json.loads(message)
        market_id, market_sid, timezone = await self.market.participant_connected(client_data)
        # async def participant_connected(self, client_data):

        self.client.publish('/'.join([self.market.market_id, client_data['id'], 'market_info']),
                            {'id': market_id,
                             'sid': market_sid,
                             'timezone': timezone, },
                            user_property=('to', client_data['sid']))

    async def on_is_market_online(self):
        self.client.publish('/'.join([self.market.market_id, 'simulation', 'market_online']), '')
        # await self.market.market_is_online()

    async def on_start_round(self, message):
        message = json.loads(message)
        await self.market.step(message['duration'], sim_params=message)

    async def on_start_generation(self, message):
        # message = json.loads(message)
        table_name = str(message) + '_' + self.market.market_id
        await self.market.open_db(table_name)

    async def on_end_generation(self, message):
        # await self.market.end_sim_generation()
        await self.market.record_transactions(delay=False)
        await self.market.ensure_transactions_complete()
        # if not last_generation:
        await self.market.reset_market()
        self.client.publish('/'.join([self.market.market_id, 'simulation', 'market_ready']), '')

    async def on_end_simulation(self):
        self.market.run = False
        # await self.market.end_sim_generation()
        await self.market.record_transactions(delay=False)
        await self.market.ensure_transactions_complete()
        await asyncio.sleep(5)
        await self.client.disconnect()
        # print('attempting to end')
        os.kill(os.getpid(), signal.SIGINT)
        raise SystemExit

    async def send_settled(self, participant_id, participant_sid, message):
        self.client.publish('/'.join([self.market.market_id, participant_id, 'settled']), message,
                            user_property=('to', participant_sid))

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
