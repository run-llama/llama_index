from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any, Optional

from llama_index.callbacks.base import BaseCallbackHandler
from llama_index.callbacks.schema import (
    CBEvent,
    CBEventType,
    EventStats,
    TIMESTAMP_FORMAT,
)

EVENTS = [
    CBEventType.LLM,
    CBEventType.EMBEDDING,
    CBEventType.CHUNKING,
    CBEventType.RETRIEVE,
    CBEventType.SYNTHESIZE,
    CBEventType.TREE,
    CBEventType.QUERY,
]


class WandbCallbackHandler(BaseCallbackHandler):
    """Callback handler that logs events to wandb.

    NOTE: this is a beta feature. The usage within our codebase, and the interface
    may change.

    This handler simply keeps track of event starts/ends, separated by event types.
    You can use this callback handler to keep track of and debug events.

    Args:
        event_starts_to_ignore (Optional[List[CBEventType]]): list of event types to
            ignore when tracking event starts.
        event_ends_to_ignore (Optional[List[CBEventType]]): list of event types to
            ignore when tracking event ends.
    """

    def __init__(
        self,
        event_starts_to_ignore: Optional[List[CBEventType]] = None,
        event_ends_to_ignore: Optional[List[CBEventType]] = None,
    ) -> None:
        """Initialize the wandb handler."""
        try:
            import wandb
            from wandb.sdk.data_types import trace_tree

            self._wandb = wandb
            self._trace_tree = trace_tree
        except ImportError:
            raise ImportError(
                "WandbCallbackHandler requires wandb. "
                "Please install it with `pip install wandb`."
            )

        if self._wandb.run is None:
            # TODO: pass wandb args
            run = self._wandb.init(project="llama_index", settings={"silent": True})
            print("W&b run URL: ", run.url)

        self._events: Dict[CBEventType, List[CBEvent]] = defaultdict(list)
        self._events_by_id = {}
        event_starts_to_ignore = (
            event_starts_to_ignore if event_starts_to_ignore else []
        )
        event_ends_to_ignore = event_ends_to_ignore if event_ends_to_ignore else []
        super().__init__(
            event_starts_to_ignore=event_starts_to_ignore,
            event_ends_to_ignore=event_ends_to_ignore,
        )

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any
    ) -> str:
        """Store event start data by event type.

        Args:
            event_type (CBEventType): event type to store.
            payload (Optional[Dict[str, Any]]): payload to store.
            event_id (str): event id to store.

        """
        event = CBEvent(event_type, payload=payload, id_=event_id)
        self._events_by_id[event.id_] = [event]
        return event.id_

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any
    ) -> None:
        """Store event end data by event type.

        Args:
            event_type (CBEventType): event type to store.
            payload (Optional[Dict[str, Any]]): payload to store.
            event_id (str): event id to store.

        """
        event = CBEvent(event_type, payload=payload, id_=event_id)
        self._events_by_id[event.id_].append(event)

        assert len(self._events_by_id[event.id_]) == 2
        assert self._events_by_id[event.id_][0].event_type == self._events_by_id[event.id_][1].event_type

        stats = self._get_time_stats(self._events_by_id[event.id_])
        event_type = self._events_by_id[event.id_][0].event_type

        # W&B Logic
        if event_type == CBEventType.LLM:
            agent_span = self._get_llm_trace_tree(event.id_, event_type, stats)
            trace = self._trace_tree.WBTraceTree(agent_span)

            if self._wandb.run is not None:
                self._wandb.run.log({"trace": trace})

            # Clear that event from the memory
            self._events_by_id.pop(event.id_, None)

    def _get_time_stats(self, event_pair: List[CBEvent]) -> EventStats:
        """Calculate time-based stats for the given event pair."""
        start_time = datetime.strptime(event_pair[0].time, TIMESTAMP_FORMAT)
        end_time = datetime.strptime(event_pair[-1].time, TIMESTAMP_FORMAT)
        total_secs = (end_time - start_time).total_seconds()

        stats = EventStats(
            start_time=start_time,
            end_time=end_time,
            total_secs=total_secs,
            average_secs=None,
            total_count=None,
        )

        return stats
    
    def _get_llm_trace_tree(self, id_, event_type, stats):
        events = self._events_by_id[id_]
        agent_span = self._trace_tree.Span(
            name=f"{event_type}-{id_}",
            span_kind = self._trace_tree.SpanKind.LLM,
            start_time_ms=int(datetime.strptime(events[0].time, TIMESTAMP_FORMAT).second*1000),
            end_time_ms=int(datetime.strptime(events[0].time, TIMESTAMP_FORMAT).second*1000),
        )
        agent_span.add_named_result(
            inputs=events[0].payload,
            outputs=events[1].payload,
        )

        return agent_span


    def finish(self) -> None:
        """Finish the callback handler."""
        self._wandb.finish()
