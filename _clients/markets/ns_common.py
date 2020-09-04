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

class NSSimulation(socketio.AsyncClientNamespace):
    def __init__(self, market):
        super().__init__(namespace='/simulation')
        self.market = market

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