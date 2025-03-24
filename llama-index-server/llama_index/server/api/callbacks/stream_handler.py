import logging
from typing import Any, AsyncGenerator, List, Optional

from llama_index.core.workflow.handler import WorkflowHandler
from llama_index.server.api.callbacks.base import EventCallback

logger = logging.getLogger("uvicorn")


class StreamHandler:
    """
    Streams events from a workflow handler through a chain of callbacks.
    """

    def __init__(
        self,
        workflow_handler: WorkflowHandler,
        callbacks: Optional[List[EventCallback]] = None,
    ):
        self.workflow_handler = workflow_handler
        self.callbacks = callbacks or []
        self.accumulated_text = ""

    async def cancel_run(self) -> None:
        """Cancel the workflow handler."""
        await self.workflow_handler.cancel_run()

    async def stream_events(self) -> AsyncGenerator[Any, None]:
        """Stream events through the processor chain."""
        try:
            async for event in self.workflow_handler.stream_events():
                # Process the event through each processor
                for callback in self.callbacks:
                    event = await callback.run(event)
                yield event

            # After all events are processed, call on_complete for each callback
            for callback in self.callbacks:
                result = await callback.on_complete(self.accumulated_text)
                if result:
                    yield result

        except Exception:
            # Make sure to cancel the workflow on error
            await self.workflow_handler.cancel_run()
            raise

    def accumulate_text(self, text: str) -> None:
        """Accumulate text from the workflow handler."""
        self.accumulated_text += text

    @classmethod
    def from_default(
        cls,
        handler: WorkflowHandler,
        callbacks: Optional[List[EventCallback]] = None,
    ) -> "StreamHandler":
        """Create a new instance with the given workflow handler and callbacks."""
        return cls(workflow_handler=handler, callbacks=callbacks)
