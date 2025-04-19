"""
trex_core.mqtt.base
===================

ONE PLACE for all the gmqtt boilerplate every TREX client repeats.

What the backbone gives you
---------------------------
    • self.client          → gmqtt.Client (already created)
    • self.msg_queue       → asyncio.Queue for raw MQTT messages
    • run()                → connects, spawns N workers, then
                             waits forever

What you *must* provide in each subclass
----------------------------------------
    • self.SUBS            list[(topic, qos)]
                            – the topics you want auto‑subscribed
    • self.dispatch        dict[str, async handler(message_dict)]
                            – exactly the table you already use
    • the usual callbacks  on_connect, on_disconnect, on_message,
                           message_processor, process_message, etc.

Optional hook
-------------
    extra_tasks() → list[coro]
        Return any background coroutines you want the run‑loop
        to schedule (e.g. Sim‑Controller’s monitor()).

Nothing else in your existing files needs to change.
"""

import asyncio
from typing import List, Tuple, Dict, Callable, Coroutine, Any
from cuid2 import Cuid as cuid
from gmqtt import Client as MQTTClient
from abc import ABC, abstractmethod

STOP = asyncio.Event()           # reused by all TREX scripts


class BaseMQTTClient(ABC):
    """
    Minimal, explicit, no‑magic backbone.
    """

    # subclass sets these two *in __init__* once it knows market_id, etc.
    SUBS: List[Tuple[str, int]] = []
    dispatch: Dict[str, Callable[[dict], Coroutine[Any, Any, None]]] = {}

    def __init__(self, server_address: str, *, consumers: int = 4):
        self.server_address = server_address
        self.consumers = consumers
        self.client = MQTTClient(cuid(length=10).generate())
        self.msg_queue: asyncio.Queue = asyncio.Queue()

    @abstractmethod
    def on_connect(self, client, flags, rc, properties):
        """Subclass must implement."""
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

    async def extra_tasks(self) -> List[Coroutine]:
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
            if handler:                         # first match wins
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
        self.client.on_connect    = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message    = self._enqueue
        await self.client.connect(self.server_address, keepalive=60)
        await STOP.wait()                       # run until someone sets STOP

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

            for coro in await self.extra_tasks():
                tg.create_task(coro)
