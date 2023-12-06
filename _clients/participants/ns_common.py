# import socketio

class NSDefault:
    def __init__(self, participant):
        # super().__init__(namespace='')
        self.participant = participant

    # async def listen():
    #     async with aiomqtt.Client("test.mosquitto.org") as client:
    #         async with client.messages() as messages:
    #             await client.subscribe("temperature/#")
    #             async for message in messages:
    #                 if message.topic.matches("humidity/inside"):
    #                     print(f"[humidity/outside] {message.payload}")
    #                 if message.topic.matches("+/outside"):
    #                     print(f"[+/inside] {message.payload}")
    #                 if message.topic.matches("temperature/#"):
    #                     print(f"[temperature/#] {message.payload}")
    #             # async for message in messages:
                #     print(message.payload)
    async def listen(self, msg_queue):
        while True:
            msg = await msg_queue.get()
            # print(not msg_queue.empty())
            # await asyncio.sleep(1)
            # client.publish("test", "hello")
            print(msg)

    async def on_connect(self):
        # print('connected')
        await self.participant.open_profile_db()
        self.participant.server_online = True
        await self.participant.join_market()

    async def on_disconnect(self):
        self.participant.server_online = False
        self.participant.busy = True

    async def on_re_register_participant(self, message):
        await self.participant.join_market()

    async def on_update_market_info(self, market_id):
        if market_id == self.participant.market_id:
            self.participant.market_connected = True

    async def on_start_round(self, message):
        await self.participant.start_round(message)

    async def on_ask_success(self, message):
        await self.participant.ask_success(message)

    async def on_bid_success(self, message):
        await self.participant.bid_success(message)

    async def on_settled(self, message):
        return await self.participant.settle_success(message)

    async def on_return_extra_transactions(self, message):
        await self.participant.update_extra_transactions(message)

# class NSSimulation(socketio.AsyncClientNamespace):
#     def __init__(self, participant):
#         super().__init__(namespace='/simulation')
#         self.participant = participant

    # async def on_connect(self):
    #     # print('connected to simulation')
    #     # await market.register()
    #     pass

    # async def on_disconnect(self):
    #     # market.server_online = False
    #     print('disconnected from simulation')
    # async def on_got_remote_actions(self, message):
    #     self.participant.trader.next_actions = message
    #     self.participant.trader.wait_for_actions.set()

    # async def on_re_register_participant(self, message):
    #     await self.participant.join_market()

    # async def on_update_curriculum(self, message):
    #     # if 'learning' in message:
    #     #     if hasattr(self.participant.trader, 'learning'):
    #     #         self.participant.trader.learning = message['learning']
    #     #
    #     # if 'exploration_factor' in message:
    #     #     if hasattr(self.participant.trader, 'exploration_factor'):
    #     #         self.participant.trader.exploration_factor = message['exploration_factor']
    #
    #     if 'anneal' in message:
    #         if hasattr(self.participant.trader, 'anneal'):
    #             anneal = message.pop('anneal')
    #             for parameter in anneal:
    #                 self.participant.trader.anneal(parameter, *anneal[parameter])
    #
    #     # set parameters
    #     for parameter in message:
    #         if hasattr(self.participant.trader, parameter):
    #             setattr(self.participant.trader, parameter, message[parameter])
    #
    # async def on_load_weights(self, message):
    #     """Event triggers loading weights for trader
    #
    #     Args:
    #         message ([type]): [description]
    #     """
    #     if hasattr(self.participant.trader, 'load_weights'):
    #         if self.participant.trader.status['weights_loading']:
    #             return
    #
    #         weights_loaded = await self.participant.trader.load_weights(**message)
    #     else:
    #         weights_loaded = True
    #
    #     if weights_loaded:
    #         await self.participant.client.emit(event='participant_weights_loaded',
    #                                            data={self.participant.participant_id: True})

    async def on_start_generation(self, message):
        """Event triggers actions to be taken before the start of a simulation

        Args:
            message ([type]): [description]
        """
        self.participant.reset()
        if hasattr(self.participant, 'storage'):
            self.participant.storage.reset(soc_pct=0)
        # self.participant.trader.output_path = message['output_path']

        if hasattr(self.participant.trader, 'metrics') and self.participant.trader.track_metrics:
            table_name = str(message['generation']) + '_' + message['market_id'] + '_metrics'
            self.participant.trader.metrics.update_db_info(message['db_string'], table_name)

    async def on_end_generation(self, message):
        """Event triggers actions to be taken at the end of a simulation

        Args:
            message ([type]): [description]
        """
        if hasattr(self.participant.trader, 'end_of_generation_tasks'):
            await self.participant.trader.end_of_generation_tasks()

        if hasattr(self.participant.trader, 'metrics') and self.participant.trader.track_metrics:
            await self.participant.trader.metrics.save()
            self.participant.trader.metrics.reset()

        # # TODO: save model
        # if hasattr(self.participant.trader, 'save_model'):
        #     await self.participant.trader.save_weights(**message)

        if hasattr(self.participant.trader, 'reset'):
            await self.participant.trader.reset(**message)

        await self.participant.client.emit(event='participant_ready',
                                           data={self.participant.participant_id: True})

    async def on_end_simulation(self, message):
        """Event tells the participant that it can terminate itself when ready.
        """
        self.participant.run = False