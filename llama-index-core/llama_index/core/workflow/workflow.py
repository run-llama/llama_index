import asyncio


from .decorators import step
from .events import StartEvent, EndEvent
from .utils import get_steps_from_class


class Workflow:
    def __init__(self, timeout: int = 10) -> None:
        """Initialize a workflow object.

        Args:
            timeout: time in second after which the workflow will exit whether or not all the steps were completed
        """
        self._timeout = timeout
        self._queues = {}
        self._tasks = set()

        self.prepare()
        self.loop = None
        self.events = []

    def prepare(self):
        for name, step in get_steps_from_class(self):
            self._queues[name] = asyncio.Queue()

            async def _task(queue, meth, target_events):
                while True:
                    ev = await queue.get()
                    if type(ev) != target_event:
                        continue

                    new_ev = await meth(ev)
                    if new_ev is None:
                        break

                    self.send_event(new_ev)

            t = asyncio.create_task(
                _task(self._queues[name], step, getattr(step, "__target_events"))
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
