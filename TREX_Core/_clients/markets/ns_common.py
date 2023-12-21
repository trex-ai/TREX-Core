import socketio

class NSDefault(socketio.AsyncClientNamespace):
    def __init__(self, market):
        super().__init__(namespace='')
        self.market = market

    # async def on_connect(self):
    #     pass

    # async def on_disconnect(self):
        # print('disconnected from server')

    # async def on_mode_switch(self, mode_data):
        # in RT mode, market controls timing
        # in sim mode, market gives up timing control to sim controller
        # self.market.mode_switch(mode_data)
        # print(mode_data)

    async def on_connect(self):
        await self.market.register()

    async def on_disconnect(self):
        self.market.server_online = False

    async def on_participant_connected(self, client_data):
        return await self.market.participant_connected(client_data)

    async def on_participant_disconnected(self, client_id):
        return await self.market.participant_disconnected(client_id)

    async def on_bid(self, bid):
        return await self.market.submit_bid(bid)

    async def on_ask(self, ask):
        return await self.market.submit_ask(ask)

    async def on_settlement_delivered(self, commit_id):
        await self.market.settlement_delivered(commit_id)

    async def on_meter_data(self, message):
        await self.market.meter_data(message)

# class NSSimulation(socketio.AsyncClientNamespace):
#     def __init__(self, market):
#         super().__init__(namespace='/simulation')
#         self.market = market

    # async def on_connect(self):
        # print('connected to simulation')
        # pass

    # async def on_disconnect(self):
        # print('disconnected from simulation')


    async def on_start_round(self, message):
        await self.market.step(message['duration'], sim_params=message)

    async def on_start_generation(self, message):
        table_name = str(message['generation']) + '_' + message['market_id']
        await self.market.open_db(message['db_string'], table_name)

    async def on_end_generation(self, message):
        await self.market.end_sim_generation()

    async def on_end_simulation(self, message):
        self.market.run = False