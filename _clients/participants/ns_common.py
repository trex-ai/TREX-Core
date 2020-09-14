import socketio

class NSDefault(socketio.AsyncClientNamespace):
    def __init__(self, participant):
        super().__init__(namespace='')
        self.participant = participant

    async def on_connect(self):
        print('connected')
        await self.participant.open_profile_db()
        self.participant.server_online = True

    async def on_disconnect(self):
        self.participant.server_online = False
        self.participant.busy = True

class NSSimulation(socketio.AsyncClientNamespace):
    def __init__(self, participant):
        super().__init__(namespace='/simulation')
        self.participant = participant

    # async def on_connect(self):
    #     # print('connected to simulation')
    #     # await market.register()
    #     pass

    # async def on_disconnect(self):
    #     # market.server_online = False
    #     print('disconnected from simulation')
    async def on_got_remote_action(self, message):
        self.participant.trader.next_actions = message
        self.participant.trader.wait_for_actions.set()

    async def on_re_register_participant(self, message):
        await self.participant.join_market()

    async def on_load_weights(self, message):
        """Event triggers loading weights for trader

        Args:
            message ([type]): [description]
        """
        if self.participant.trader.learning:
            if not hasattr(self.participant.trader, 'warmup') or self.participant.trader.warmup ^ message[
                'warm_up']:
                print('changing lrn mode')
                self.participant.trader.warmup = message['warm_up']
                if 'gen_len' in message:
                    self.participant.trader.gen_len = message['gen_len']

                if hasattr(self.participant.trader, 'set_lrn'):
                    await self.participant.trader.set_lrn()

        if hasattr(self.participant.trader, 'load_weights') and 'load_weights' in message and message['load_weights']:
            weights_loaded = await self.participant.trader.load_weights(message['db_path'],
                                                                        message['generation'],
                                                                        message['market_id'],
                                                                        message['reset'])
        else:
            weights_loaded = True

        if weights_loaded:
            await self.participant.client.emit(event='participant_weights_loaded',
                                               data={self.participant.participant_id: True},
                                               namespace='/simulation')

    async def on_start_generation(self, message):
        """Event triggers actions to be taken before the start of a simulation

        Args:
            message ([type]): [description]
        """
        self.participant.reset()
        if self.participant.storage:
            self.participant.storage.reset(soc_pct=0)
        self.participant.trader.output_path = message['output_path']

        if hasattr(self.participant.trader, 'metrics') and self.participant.trader.track_metrics:
            table_name = str(message['generation']) + '_' + message['market_id'] + '_metrics'
            self.participant.trader.metrics.update_db_info(message['db_string'], table_name)

        # if hasattr(self.participant.trader, 'replay'):
        #     table_name = str(message['generation']) + '_' + 'validation' + '_metrics'
        #     self.participant.trader.replay.update_db_info(message['db_string'], table_name)

    async def on_end_generation(self, message):
        """Event triggers actions to be taken at the end of a simulation

        Args:
            message ([type]): [description]
        """
        if hasattr(self.participant.trader, 'metrics') and self.participant.trader.track_metrics:
            await self.participant.trader.metrics.save()
            self.participant.trader.metrics.reset()

        self.participant.trader.status['weights_loaded'] = False
        if self.participant.trader.learning:
            if hasattr(self.participant.trader, 'save_weights'):
                weights_saved = await self.participant.trader.save_weights(**message)
        await self.participant.trader.reset(**message)

        await self.participant.client.emit(event='participant_weights_saved',
                                           data={self.participant.participant_id: True},
                                           namespace='/simulation')

    async def on_end_simulation(self, message):
        """Event tells the participant that it can terminate itself when ready.
        """
        self.participant.run = False