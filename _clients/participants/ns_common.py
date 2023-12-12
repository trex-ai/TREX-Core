# import socketio

class NSDefault:
    def __init__(self, participant):
        # super().__init__(namespace='')
        self.participant = participant

    async def listen(self, msg_queue):
        while self.participant.run:
            msg = await msg_queue.get()
            topic_event = msg['topic'].split('/')[-1]
            payload = msg['payload']

            match topic_event:
                # market related events
                case 'update_market_info':
                    await self.on_update_market_info(payload)
                case 'start_round':
                    await self.on_start_round(payload)
                case 'ask_success':
                    await self.on_ask_success(payload)
                case 'bid_success':
                    await self.on_bid_success(payload)
                case 'settled':
                    await self.on_settled(payload)
                case 'return_extra_transactions':
                    await self.on_return_extra_transactions(payload)
                # simulation related events
                case 're_register_participant':
                    await self.on_re_register_participant(payload)
                case 'start_generation':
                    await self.on_start_generation(payload)
                case 'end_generation':
                    await self.on_end_generation(payload)
                case 'end_simulation':
                    await self.on_end_simulation()
        else:
            await self.participant.kill()
            # return True
            # print(msg)

    async def on_connect(self):
        # print('connected')
        await self.participant.open_profile_db()
        self.participant.server_online = True
        await self.participant.join_market()

    # async def on_disconnect(self):
    def on_disconnect(self):
        self.participant.server_online = False
        self.participant.busy = True

    async def on_update_market_info(self, market_id):
        if market_id == self.participant.market_id:
            self.participant.market_connected = True
            # self.participant.busy = False

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

    async def on_re_register_participant(self, message):
        await self.participant.join_market()
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

    async def on_end_simulation(self):
        """Event tells the participant that it can terminate itself when ready.
        """
        self.participant.run = False
