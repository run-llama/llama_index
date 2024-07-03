import asyncio
import warnings

from .decorators import step
from .events import StartEvent, StopEvent, Event, EventType
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
        self._events = []

        self.prepare()
        self.loop = None

    def prepare(self):
        for name, step in get_steps_from_class(self):
            self._queues[name] = asyncio.Queue()

            async def _task(name, queue, step, target_events):
                while True:
                    ev = await queue.get()
                    if type(ev) not in target_events:
                        continue

                    new_ev = await step(ev)
                    if new_ev is None:
                        continue

                    if not isinstance(new_ev, Event):
                        warnings.warn(
                            f"Step function {name} didn't return an Event instance. Returned value: {new_ev}"
                        )
                    else:
                        self.send_event(new_ev)

            self._tasks.add(
                asyncio.create_task(
                    _task(
                        name, self._queues[name], step, getattr(step, "__target_events")
                    )
                )
            )

    def send_event(self, message):
        for queue in self._queues.values():
            queue.put_nowait(message)
        self._events.append(message)

    async def run(self, **kwargs):
        self._events = []
        async with asyncio.timeout(self.timeout):
            self.send_event(StartEvent(kwargs))
            try:
                await asyncio.gather(*list(self.tasks))
            except asyncio.CancelledError:
                pass

    @step(StopEvent)
    async def done(self, _: EventType):
        """Tears down the whole workflow and stop execution."""
        for t in self._tasks:
            t.cancel()
        print("Broker log:")
        print("\n".join(str(type(ev).__name__) for ev in self._events))
