import asyncio
import warnings

from .decorators import step
from .events import StartEvent, StopEvent, Event
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
        self._retval = None

    def _start(self):
        for name, step in get_steps_from_class(self).items():
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
        """Dispatches an event to all the queues available."""
        for queue in self._queues.values():
            queue.put_nowait(message)
        self._events.append(message)

    async def run(self, **kwargs):
        """Entrypoint for every workflow.

        The user input is wrapped into a StartEvent that's dispatched to initiate
        the workflow
        """
        self._events = []
        if not self._tasks:
            self._start()

        async with asyncio.timeout(self._timeout):
            self.send_event(StartEvent(kwargs))
            try:
                await asyncio.gather(*list(self._tasks))
            except asyncio.CancelledError:
                pass

            return self._retval

    @step(StopEvent)
    async def done(self, ev: StopEvent):
        """Tears down the whole workflow and stop execution."""
        # Stop all the tasks
        for t in self._tasks:
            t.cancel()
        # Remove any reference to the tasks
        self._tasks = set()
        self._retval = ev.msg or None
