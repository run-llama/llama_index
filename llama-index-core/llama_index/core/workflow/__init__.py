import asyncio
import inspect
from collections import UserDict


class Event:
    def __init__(self, *args, **kwargs) -> None:
        pass


class StartEvent(UserDict, Event):
    pass


class EndEvent(Event):
    pass


def step(fn):
    sig = inspect.signature(fn)
    param = sig.parameters.get("ev")
    if param is None:
        raise ValueError("The method must accept an event")
    target_event = param.annotation
    fn.__target_events = [target_event]
    return fn


class Workflow:
    def __init__(self, timeout=10) -> None:
        self.timeout = timeout
        self.queues = {}
        self.tasks = set()
        self.prepare()
        self.loop = None
        self.events = []

    def prepare(self):
        meths = inspect.getmembers(self, predicate=inspect.ismethod)
        for name, meth in meths:
            if not hasattr(meth, "__target_events"):
                continue
            self.queues[name] = asyncio.Queue()

            async def _task(queue, meth, target_event):
                while True:
                    ev = await queue.get()
                    if type(ev) != target_event:
                        continue

                    new_ev = await meth(ev)
                    if new_ev is None:
                        break

                    self.send_event(new_ev)

            t = asyncio.create_task(
                _task(self.queues[name], meth, getattr(meth, "__target_events")[0])
            )
            self.tasks.add(t)

    def subscribe(self):
        task_id = len(self.queues) + 1
        self.queues[task_id] = asyncio.Queue()
        return task_id

    def unsubscribe(self, task_id):
        del self.queues[task_id]

    def send_event(self, message):
        self.events.append(message)
        for queue in self.queues.values():
            queue.put_nowait(message)

    async def run(self, **kwargs):
        async with asyncio.timeout(self.timeout):
            self.send_event(StartEvent(kwargs))
            try:
                await asyncio.gather(*list(self.tasks))
            except asyncio.CancelledError:
                pass

    @step
    async def done(self, ev: EndEvent):
        for t in self.tasks:
            t.cancel()
        print("Broker log:")
        print("\n".join(str(type(ev).__name__) for ev in self.events))
