import asyncio
from typing import Any, AsyncGenerator, Optional

from llama_index.core.workflow.context import Context
from llama_index.core.workflow.events import Event, StopEvent
from llama_index.core.workflow.errors import WorkflowDone


class WorkflowHandler(asyncio.Future):
    def __init__(
        self, *args: Any, ctx: Optional[Context] = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.ctx = ctx

    def __str__(self) -> str:
        return str(self.result())

    def is_done(self) -> bool:
        return self.done()

    async def stream_events(self) -> AsyncGenerator[Event, None]:
        if not self.ctx:
            raise ValueError("Context is not set!")

        while True:
            ev = await self.ctx.streaming_queue.get()

            yield ev

            if type(ev) is StopEvent:
                break

    async def run_step(self) -> Optional[Any]:
        if self.ctx and not self.ctx.stepwise:
            raise ValueError("Stepwise context is required to run stepwise.")

        if self.ctx:
            # Unblock all pending steps
            for flag in self.ctx._step_flags.values():
                flag.set()

            # Yield back control to the event loop to give an unblocked step
            # the chance to run (we won't actually sleep here).
            await asyncio.sleep(0)

            # See if we're done, or if a step raised any error
            we_done = False
            exception_raised = None
            for t in self.ctx._tasks:
                # Check if we're done
                if not t.done():
                    continue

                we_done = True
                e = t.exception()
                if type(e) != WorkflowDone:
                    exception_raised = e

            retval = None
            if we_done:
                # Remove any reference to the tasks
                for t in self.ctx._tasks:
                    t.cancel()
                    await asyncio.sleep(0)
                retval = self.ctx.get_result()

                self.set_result(retval)

            if exception_raised:
                raise exception_raised
        else:
            raise ValueError("Context is not set!")

        return retval
