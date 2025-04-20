import asyncio
from typing import List, Tuple, Dict, Callable, Coroutine, Any
from cuid2 import Cuid
from gmqtt import Client as MQTTClient
from abc import ABC, abstractmethod

# STOP = asyncio.Event()           # reused by all TREX scripts


class BaseMQTTClient(ABC):
    SUBS: List[Tuple[str, int]] = []
    dispatch: Dict[str, Callable[[dict], Coroutine[Any, Any, None]]] = {}

    def __init__(self, server_address: str, port: int = 1883, consumers: int = 1):
        self.cuid = Cuid(length=10).generate()
        self.server_address = server_address
        self.port = port
        self.consumers = consumers
        self.client = MQTTClient(self.cuid)
        self.msg_queue: asyncio.Queue = asyncio.Queue()

    @abstractmethod
    def on_connect(self, client, flags, rc, properties):
        raise NotImplementedError

    @abstractmethod
    def on_disconnect(self, client, packet, exc=None):
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Helpers the subclass *may* call / override
    # ------------------------------------------------------------------ #
    def subscribe_common(self, client: MQTTClient) -> None:
        """Call inside your own on_connect to subscribe everything in SUBS."""
        for topic, qos in self.SUBS:
            client.subscribe(topic, qos=qos)

    async def background_tasks(self) -> List[Coroutine]:
        """
        Subclass can override to return extra background coroutines
        that run alongside the MQTT loop (e.g. controller.monitor()).
        """
        return []

    # ------------------------------------------------------------------ #
    # Internal: queue raw messages
    # ------------------------------------------------------------------ #
    async def _enqueue(self, client, topic, payload, qos, properties):
        await self.msg_queue.put(
            {"topic": topic, "payload": payload.decode(), "properties": properties}
        )

    # ------------------------------------------------------------------ #
    # Internal: dispatch loop (uses self.dispatch exactly like originals)
    # ------------------------------------------------------------------ #
    async def _dispatch(self, message: dict) -> None:
        for segment in reversed(message["topic"].split("/")):
            handler = self.dispatch.get(segment)
            if handler:
                await handler(message)
                return
        print("unrecognised topic:", message["topic"])

    async def _message_processor(self) -> None:
        while True:
            msg = await self.msg_queue.get()
            try:
                await self._dispatch(msg)
            finally:
                self.msg_queue.task_done()

    # ------------------------------------------------------------------ #
    # Internal: connect + stay alive
    # ------------------------------------------------------------------ #
    async def _connect_forever(self) -> None:
        """
        Uses the subclass's *own* on_connect/on_disconnect implementations.
        """
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self._enqueue
        await self.client.connect(self.server_address, self.port, keepalive=60)
        # await STOP.wait()

    # ------------------------------------------------------------------ #
    # Public: start everything
    # ------------------------------------------------------------------ #
    async def run(self) -> None:
        """
        Call this from your script's `if __name__ == "__main__":`
        instead of rolling your own TaskGroup.
        """
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._connect_forever())

            for _ in range(self.consumers):     # message queue workers
                tg.create_task(self._message_processor())

            for coro in await self.background_tasks():
                tg.create_task(coro)
