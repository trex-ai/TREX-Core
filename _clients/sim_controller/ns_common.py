import socketio

class NSDefault(socketio.AsyncClientNamespace):
    def __init__(self, controller):
        super().__init__(namespace='')
        self.controller = controller

    # async def on_connect(self):
    #     await self.controller.register()

    # async def on_disconnect(self):
        # pass
