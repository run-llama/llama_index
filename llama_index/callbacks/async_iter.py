import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Union, cast

from llama_index.callbacks.base_handler import BaseCallbackHandler
from llama_index.callbacks.schema import CBEventType
from llama_index.callbacks.utils import superjson_dumps

logger = logging.getLogger(__name__)


class AsyncIteratorCallbackHandler(BaseCallbackHandler):
    """Callback handler that returns an async iterator."""

    queue: asyncio.Queue[str]
    done: asyncio.Event

    def __init__(
        self,
        event_starts_to_ignore: Optional[List[str]] = [],
        event_ends_to_ignore: Optional[List[str]] = [],
        verbose: Optional[bool] = True,
    ) -> None:
        super().__init__(
            event_starts_to_ignore=event_starts_to_ignore,
            event_ends_to_ignore=event_ends_to_ignore,
        )
        self.queue = asyncio.Queue()
        self.done = asyncio.Event()
        self.verbose = verbose

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Run when an event starts and return id of event."""
        event = {
            "event": event_type + ".start",
            "event_id": event_id,
            "parent_id": parent_id,
        }
        print(event)
        event = superjson_dumps(event)
        print(event)
        if self.verbose:
            logger.debug(
                event_type,
                payload=payload,
                event_id=event_id,
                parent_id=parent_id,
                **kwargs,
            )
        self.queue.put_nowait(event)

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Run when an event ends."""
        event = {
            "event": event_type + ".end",
            "event_id": event_id,
        }
        if event_type not in [CBEventType.AGENT_STEP]:
            event["data"] = payload
        print(event)
        event = superjson_dumps(event)
        print(event)
        if self.verbose:
            logger.debug(
                event_type,
                # payload=payload,
                # event_id=event_id,
                # **kwargs,
            )
        self.queue.put_nowait(event)

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Run when an overall trace is launched."""
        event = {
            "event": "trace.start",
            "trace_id": trace_id,
        }
        if self.verbose:
            logger.debug(
                "trace.start",
                trace_id=trace_id,
            )
        self.queue.put_nowait(superjson_dumps(event))

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Run when an overall trace is exited."""
        event = {
            "event": "trace.end",
            "trace_id": trace_id,
            "trace_map": trace_map,
        }
        if self.verbose:
            logger.debug(
                "trace.start",
                trace_id=trace_id,
                trace_map=trace_map,
            )
        self.queue.put_nowait(superjson_dumps(event))
        self.done.set()

    async def aiter(self) -> AsyncIterator[str]:
        while not self.queue.empty() or not self.done.is_set():
            # Wait for the next token in the queue,
            # but stop waiting if the done event is set
            done, other = await asyncio.wait(
                [
                    # NOTE: If you add other tasks here, update the code below,
                    # which assumes each set has exactly one task each
                    asyncio.ensure_future(self.queue.get()),
                    asyncio.ensure_future(self.done.wait()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel the other task
            if other:
                other.pop().cancel()

            # Extract the value of the first completed task
            token_or_done = cast(Union[str, Literal[True]], done.pop().result())

            # If the extracted value is the boolean True, the done event was set
            if token_or_done is True:
                break

            # Otherwise, the extracted value is a token, which we yield
            yield token_or_done
