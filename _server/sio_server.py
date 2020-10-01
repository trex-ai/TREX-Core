"""Socket.io server for TREX

Main functions are client management and message relay
"""

import sys
sys.path.append("C:/source/TREX-Core")
import os
import asyncio
import socket
import socketio
from _utils import jkson as json

if os.name == 'posix':
    import uvloop
    uvloop.install()

from aiohttp import web

server = socketio.AsyncServer(async_mode='aiohttp', json=json)
app = web.Application()
server.attach(app)

# clients are grouped by market ID. This means that only 1 market can exist per group
# sessions and clients are used to keep track of client states.
sessions = {}
clients = {}

async def send_market_info(market_id, client_sid):
    """Callback function for when the market client successfully received the join request from participant.

    Args:
        market_id (string): Market ID
        client_sid: client session ID
    """
    if client_sid and client_sid in sessions:
        server.enter_room(client_sid, market_id, namespace='/market')

        market_id = sessions[client_sid]['market_id']
        await server.emit(event='update_market_info',
                          data=market_id,
                          to=client_sid,
                          namespace='/market')

        await server.emit(event='participant_joined',
                          data=sessions[client_sid]['client_id'],
                          namespace='/simulation',
                          room=market_id)

class Default(socketio.AsyncNamespace):
    async def on_connect(self, sid, environ):
        pass
        # print('Client connected')
        # clients[sid] = {}
        # print(clients)
        # print('Client', sid, 'connected')

    # Disconnect client from server
    async def on_disconnect(self, sid):
        client = sessions.pop(sid, None) if sid in sessions else None
        if client:
            print(client, sid, 'disconnected')
            # remove participant from participant and market rooms
            if client['client_type'] == 'participant':
                client_id = client['client_id']
                market_id = client['market_id']
                clients[market_id]['participant'][client_id]['online'] = False

                # if sim controller exists, notify sim controller that a participant has disconnected
                if 'sim_controller' in clients[market_id]:
                    await server.emit(
                        event='participant_disconnected',
                        data=client_id,
                        to=clients[market_id]['sim_controller']['sid'],
                        namespace='/simulation')

class ETXMarket(socketio.AsyncNamespace):
    async def on_connect(self, sid, environ):
        self.settle_buf = {}
        # print(sessions)
        pass
    # Register market in server

    async def on_register(self, sid, client_data):
        """Event emitted by market module to register itself as a client

        Args:
            sid ([type]): [description]
            client_data ([type]): [description]

        Returns:
            [type]: [description]
        """
        if client_data['type'][0] == 'market':
            # create a market_id group if one does not exist in clients
            market_id = client_data['id']
            if market_id not in clients:
                clients[market_id] = {}

            server.enter_room(sid, market_id, namespace='/market')

            clients[market_id] = {}
            clients[market_id]['market'] = {
                'type': client_data['type'][1],
                'id': client_data['id'],
                'sid': sid
            }
            sessions[sid] = {'client_id': market_id,
                             'client_type': 'market',
                             'market_id': market_id}

            return True

    async def on_join(self, sid, client_data):
        """Event emitted by participant to join a market

        Args:
            sid ([type]): [description]
            client_data ([type]): [description]

        Returns:
            [type]: [description]
        """
        market_id = client_data['market_id']
        if market_id not in clients:
            return False

        #id cannot be empty
        if not client_data['id']:
            return False

        # Register client in server
        if client_data['type'][0] == 'participant':
            # Add client to session dict
            sessions[sid] = {'client_id': client_data['id'],
                             'client_type': 'participant',
                             'market_id': market_id}

            # Add client to client list
            if 'participant' not in clients[market_id]:
                clients[market_id]['participant'] = {}

            clients[market_id]['participant'][client_data['id']] = {
                'online': True,
                'sid': sid
            }

            c_data = {
                'type': client_data['type'][1],
                'id': client_data['id'],
                'sid': sid
            }
            # Register client in server
            server.enter_room(sid, market_id, namespace='/simulation')
            server.enter_room(clients[market_id]['market']['sid'], market_id, namespace='/simulation')

            await server.emit(event='participant_connected',
                              data=c_data,
                              namespace='/market',
                              to=clients[market_id]['market']['sid'],
                              callback=send_market_info)
            return True

    async def on_start_round(self, sid, message):
        """Event emitted by market start the next round
        Also signals the end of the current round in real-time mode

        Args:
            sid ([type]): [description]
            message ([type]): [description]
        """
        await server.emit(
            event='start_round',
            data=message,
            room=sessions[sid]['market_id'],
            namespace='/market')

    async def on_bid(self, sid, bid):
        """Event emitted by participant to submit a bid

        Args:
            sid ([type]): [description]
            bid ([type]): [description]
        """

        async def bid_cb(buyer_sid, message):
            """Callback function to notify participant that the bid submission has been relayed to the Market

            Args:
                buyer_sid ([type]): [description]
                message ([type]): [description]
            """
            if message['uuid'] is not None:
                await server.emit('bid_success', message, namespace='/market', room=buyer_sid)

        bid['session_id'] = sid
        market_id = sessions[sid]['market_id']
        market_sid = clients[market_id]['market']['sid']
        await server.emit(
            event='bid',
            data=bid,
            to=market_sid,
            namespace='/market',
            callback=bid_cb)

    async def on_ask(self, sid, ask):
        """Event emitted by participant to submit an ask

        Args:
            sid ([type]): [description]
            ask ([type]): [description]
        """
        async def ask_cb(seller_sid, message):
            """Callback function to notify participant that the ask submission has been relayed to the Market

            Args:
                seller_sid ([type]): [description]
                message ([type]): [description]
            """
            if message['uuid'] is not None:
                await server.emit('ask_success', message, namespace='/market', room=seller_sid)

        ask['session_id'] = sid
        market_id = sessions[sid]['market_id']
        market_sid = clients[market_id]['market']['sid']
        await server.emit(
            event='ask',
            data=ask,
            to=market_sid,
            namespace='/market',
            callback=ask_cb)

    # Process successful settlements
    # event emitted by market
    async def on_send_settlement(self, sid, message):
        """Event emitted by market to notify both parties of a successful settlement

        Args:
            sid ([type]): [description]
            message ([type]): [description]
        """
        async def settled_cb(commit_id):
            """Callback function to notify the market that the settlement has been relayed

            Args:
                commit_id ([type]): [description]
            """
            self.settle_buf[commit_id] ^= True
            if self.settle_buf[commit_id]:
                self.settle_buf.pop(message['commit_id'], None)
                await server.emit(event='settlement_delivered', data=commit_id, to=market_sid, namespace='/market')

        buyer = message['buyer_id']
        seller = message['seller_id']
        market_id = sessions[sid]['market_id']
        market_sid = clients[market_id]['market']['sid']

        if buyer == 'grid' or seller == 'grid':
            return

        # Make sure participants in settlement are connected to server
        buyer_online = clients[market_id]['participant'][buyer]['online']
        seller_online = clients[market_id]['participant'][seller]['online']
        buyer_sid = clients[market_id]['participant'][buyer]['sid']
        seller_sid = clients[market_id]['participant'][seller]['sid']

        if buyer_online and seller_online:
            self.settle_buf[message['commit_id']] = True
            await server.emit(event='settled', data=message, to=buyer_sid, namespace='/market', callback=settled_cb)
            await server.emit(event='settled', data=message, to=seller_sid, namespace='/market', callback=settled_cb)
        else:
            await server.emit(event='settlement_delivered', data=message['commit_id'], to=market_sid, namespace='/market')

    async def on_meter_data(self, sid, meter):
        """Event emitted by participant to provide the Market with the submetering data for the round that just ended.

        Args:
            sid ([type]): [description]
            meter ([type]): [description]
        """
        if sid not in sessions:
            return
        message = {
            'participant_id': sessions[sid]['client_id'],
            'meter': meter
        }

        market_id = sessions[sid]['market_id']
        market_sid = clients[market_id]['market']['sid']
        await server.emit(event='meter_data',
                          data=message,
                          to=market_sid,
                          namespace='/market')

    # event emitted by market
    async def on_return_extra_transactions(self, sid, transactions):
        """Event emitted by the Market to notify the participants of the extra, (financial and grid) transactions incurred after the delivery allocation.

        Args:
            sid ([type]): [description]
            transactions ([type]): [description]
        """
        message = transactions
        participant_id = message.pop('participant')
        market_id = sessions[sid]['market_id']

        if participant_id not in clients[market_id]['participant']:
            return

        participant_sid = clients[market_id]['participant'][participant_id]['sid']
        if participant_sid not in sessions:
            return

        await server.emit(event='return_extra_transactions',
                          data=message,
                          room=participant_sid,
                          namespace='/market')
import random
class Simulation(socketio.AsyncNamespace):
    # def __init__(self):
    #     super().__init__(namespace='/simulation')
    #         self.market = market
    #     pass

    # async def on_connect(self, sid, environ):
    # #     # print('connect to sim')
    #     pass

    async def on_remote_agent_status(self, sid, data):
        print('Server: remote agent status')
        if not clients['remote_agent']['sid']:
            return
        market_id = data['market_id']
        if clients[market_id]['sim_controller']['sid'] == sid:
            await server.emit(
                event='remote_agent_status',
                data=data,
                to=clients['remote_agent']['sid'],
                namespace='/simulation')

    async def on_remote_agent_ready(self, sid, market_id):
        print('server: remote agent ready ')
        if clients['remote_agent']['sid'] == sid:
            await server.emit(
                event='remote_agent_ready',
                room=market_id['market_id'],
                namespace='/simulation')

    async def on_get_remote_actions(self, sid, observations):
        """Event emitted by the thin remote agent to get next actions from a centralized learning agent
        Args:
            sid ([type]): [description]
            observations ([type]): [description]
        """
        # await server.sleep(random.randint(1, 3))
        # if not 'remote_agent' in clients:
        #     await server.emit(
        #         event='got_remote_actions',
        #         data={},
        #         to=sid,
        #         namespace='/simulation')
        # else:
        await server.emit(
            event='get_remote_actions',
            data=observations,
            to=clients['remote_agent']['sid'],
            namespace='/simulation')

    async def on_got_remote_actions(self, sid, actions):
        """Event emitted by the centralized learning agent to get return actions to the thin remote agent
        Args:
            sid ([type]): [description]
            actions ([type]): [description]
        """
        participant_id = actions.pop('participant_id')
        market_id = actions.pop('market_id')
        participant_sid = clients[market_id]['participant'][participant_id]['sid']
        if participant_sid not in sessions:
            return

        await server.emit(event='got_remote_actions',
                          data=actions,
                          room=participant_sid,
                          namespace='/simulation')

    async def on_register(self, sid, client_data):
        """Event emitted by the simulation controller to register itself on the server

        Args:
            sid ([type]): [description]
            client_data ([type]): [description]

        Returns:
            [type]: [description]
        """
        if client_data['type'][0] == 'sim_controller':
            market_id = client_data['market_id']

            # Confirm target market exists
            if market_id not in clients:
                return False
            else:
                # register sim controller in session and client lists
                sessions[sid] = {'client_id': client_data['id'],
                                 'client_type': 'sim_controller',
                                 'market_id': market_id}

                clients[market_id]['sim_controller'] = {
                    'type': client_data['type'][1],
                    'id': client_data['id'],
                    'sid': sid
                }
                # register sim controller in server
                server.enter_room(sid, market_id, namespace='/market')
                server.enter_room(sid, market_id, namespace='/simulation')
                return True

        elif client_data['type'][0] == 'remote_agent':
            # FIXME: make sure that you name this to what the gym agent needs to be
            # register sim controller in session and client lists
            # sessions[sid] = {'client_id': client_data['id'],
            #                  'client_type': 'remote_agent'}

            clients['remote_agent'] = {
                'sid': sid
            }
        return False



    async def on_re_register_participant(self, sid):
        """Event emitted by simulation controller to re-register all participants, in case some were missed during initialization

        Args:
            sid ([type]): [description]
        """
        market_id = sessions[sid]['market_id']
        if not 'sim_controller' in clients[market_id]:
            return

        if clients[market_id]['sim_controller']['sid'] == sid:
            await server.emit(
                event='re_register_participant',
                data='',
                room=market_id,
                namespace='/market')

    async def on_participant_weights_loaded(self, sid, message):
        """Event emitted by participant traders to notify the simulation controller that the weights have been loaded.

        Args:
            sid ([type]): [description]
            message ([type]): [description]
        """
        market_id = sessions[sid]['market_id']
        sim_controller_sid = clients[market_id]['sim_controller']['sid']

        await server.emit(
            event='participant_weights_loaded',
            data=message,
            to=sim_controller_sid,
            namespace='/simulation')

    # event emitted by participant
    async def on_participant_weights_saved(self, sid, message):
        """Event emitted by participant traders to notify the simulation controller that the weights have been saved.

        Args:
            sid ([type]): [description]
            message ([type]): [description]
        """
        market_id = sessions[sid]['market_id']
        sim_controller_sid = clients[market_id]['sim_controller']['sid']
        await server.emit(
            event='participant_weights_saved',
            data=message,
            to=sim_controller_sid,
            namespace='/simulation')

    # event emitted by sim controller
    async def on_load_weights(self, sid, message):
        """Event emitted by the sim controller to allow participant traders to load weights

        Args:
            sid ([type]): [description]
            message ([type]): [description]
        """
        market_id = sessions[sid]['market_id']
        sim_controller_sid = clients[market_id]['sim_controller']['sid']

        if sim_controller_sid == sid:
            participant_id = message.pop('participant_id', None)
            if participant_id in clients[market_id]['participant']:
                participant_sid = clients[market_id]['participant'][participant_id]['sid']
            else:
                participant_sid = None

            if participant_sid:
                await server.emit(
                    event='load_weights',
                    data=message,
                    to=participant_sid,
                    namespace='/simulation'
                )

    async def on_start_round(self, sid, message):
        """Event emitted by simulation controller to start the next round. Event is redirected to Market to trigger the start of the next round.

        Args:
            sid ([type]): [description]
            message ([type]): [description]
        """

        market_id = sessions[sid]['market_id']
        sim_controller_sid = clients[market_id]['sim_controller']['sid']
        market_sid = clients[market_id]['market']['sid']

        if sim_controller_sid == sid:
            await server.emit(
                event='start_round',
                data=message,
                to=market_sid,
                namespace='/simulation')

    async def on_end_round(self, sid, message):
        """Event emitted by market to notify sim controller that all market functions for the current round are complete.

        This is a special signal that only exists in simulation mode, as the flow of time is continuous in real-time mode.

        Args:
            sid ([type]): [description]
            message ([type]): [description]
        """
        market_id = sessions[sid]['market_id']
        sim_controller_sid = clients[market_id]['sim_controller']['sid']
        market_sid = clients[market_id]['market']['sid']

        if market_sid == sid:
            await server.emit(
                event='end_round',
                data=message,
                to=sim_controller_sid,
                namespace='/simulation')

    async def on_end_turn(self, sid):
        """Event emitted by participant to notify sim controller that it has performed all of its actions for the current round.

        This is a special signal that only exists in simulation mode, as the market does not wait to start the next round in real-time mode.

        Args:
            sid ([type]): [description]
        """
        if sid not in sessions:
            return

        market_id = sessions[sid]['market_id']
        sim_controller_sid = clients[market_id]['sim_controller']['sid']

        await server.emit(event='end_turn',
                          data=sessions[sid]['client_id'],
                          to=sim_controller_sid,
                          namespace='/simulation')


    async def on_start_generation(self, sid, message):
        """Event emitted by sim controller to notify all clients in the market of the start of a new generation.

        Mainly used to reset parameters.

        Args:
            sid ([type]): [description]
            message ([type]): [description]
        """
        market_id = sessions[sid]['market_id']
        sim_controller_sid = clients[market_id]['sim_controller']['sid']

        if sim_controller_sid == sid:
            await server.emit(
                event='start_generation',
                data=message,
                to=market_id,
                namespace='/simulation')

    async def on_end_generation(self, sid, message):
        """Event emitted by sim controller to notify all clients in the market of the end of the current generation.

        Args:
            sid ([type]): [description]
            message ([type]): [description]
        """
        market_id = sessions[sid]['market_id']
        sim_controller_sid = clients[market_id]['sim_controller']['sid']
        if sim_controller_sid == sid:
            await server.emit(
                event='end_generation',
                data=message,
                to=market_id,
                namespace='/simulation')

    # event emitted by sim controller
    # redirected to all participants
    async def on_end_simulation(self, sid):
        """Event emitted by sim controller to notify all clients that the simulation has ended, and it is safe to quit when they are ready to quit.

        Args:
            sid ([type]): [description]
        """
        market_id = sessions[sid]['market_id']
        sim_controller_sid = clients[market_id]['sim_controller']['sid']
        if sim_controller_sid == sid:
            await server.emit(
                event='end_simulation',
                data='',
                to=market_id,
                namespace='/simulation')

    async def on_is_market_online(self, sid):
        """Event emitted by sim controller to ask server if a Market has been registered

        Args:
            sid ([type]): [description]
        """
        market_id = sessions[sid]['market_id']
        if market_id in clients:
            await server.emit(
                event='market_online',
                data='',
                to=sid,
                namespace='/simulation')

    async def on_market_ready(self, sid):
        """Event emitted by the Markeet to notify the sim controller that it is ready to operate

        Args:
            sid ([type]): [description]
        """
        market_id = sessions[sid]['market_id']
        market_sid = clients[market_id]['market']['sid']
        sim_controller_sid = clients[market_id]['sim_controller']['sid']
        if market_sid == sid:
            await server.emit(
                event='market_ready',
                to=sim_controller_sid,
                namespace='/simulation')

# register namespaces
server.register_namespace(Default(''))
server.register_namespace(ETXMarket('/market'))
server.register_namespace(Simulation('/simulation'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--host', default=socket.gethostbyname(socket.getfqdn()), help='')
    parser.add_argument('--port', default=42069, help='')
    args = parser.parse_args()

    web.run_app(app=app, host=args.host, port=str(args.port))

