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
    CBEventType.SYNTHESIZE ,
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
            self._wandb = wandb
        except ImportError:
            raise ImportError(
                "WandbCallbackHandler requires wandb. "
                "Please install it with `pip install wandb`."
            )

        if self._wandb.run is None:
            self._wandb.init(project="llama_index", settings={"silent": True})
            print("W&b run URL: ", self._wandb.run.settings.run_url)

        self._events: Dict[CBEventType, List[CBEvent]] = defaultdict(list)
        self._sequential_events: List[CBEvent] = []
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
        self._events[event.event_type].append(event)
        self._sequential_events.append(event)
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
        self._events[event.event_type].append(event)
        self._sequential_events.append(event)

    def get_events(self, event_type: Optional[CBEventType] = None) -> List[CBEvent]:
        """Get all events for a specific event type."""
        return self._events[event_type]

    def _get_event_pairs(self, events: List[CBEvent]) -> List[List[CBEvent]]:
        """Helper function to pair events according to their ID."""
        event_pairs: Dict[str, List[CBEvent]] = defaultdict(list)
        for event in events:
            event_pairs[event.id_].append(event)

        sorted_events = sorted(
            event_pairs.values(),
            key=lambda x: datetime.strptime(x[0].time, TIMESTAMP_FORMAT),
        )
        return sorted_events

    def _get_time_stats(
        self, event_pair: List[CBEvent]
    ) -> EventStats:
        """Calculate time-based stats for the given event pair."""
        start_time = datetime.strptime(event_pair[0].time, TIMESTAMP_FORMAT)
        end_time = datetime.strptime(event_pair[-1].time, TIMESTAMP_FORMAT)
        total_secs = (end_time - start_time).total_seconds()

        stats = EventStats(
            start_time=start_time,
            end_time=end_time,
            total_secs=total_secs,
            average_secs=None,
            total_count=None
        )

        return stats

    def get_event_pairs(
        self, event_type: Optional[CBEventType] = None
    ) -> List[List[CBEvent]]:
        """Pair events by ID, either all events or a sepcific type."""
        if event_type is not None:
            return self._get_event_pairs(self._events[event_type])

        return self._get_event_pairs(self._sequential_events)

    def get_event_time_info_by_pair(
        self, event_pairs: List[List[CBEvent]]
    ) -> EventStats:
        return self._get_time_stats_from_event_pairs(event_pairs)

    def flush_event_logs(self) -> None:
        """Clear all events from memory."""
        self._events = defaultdict(list)
        self._sequential_events = []

    def log_events_to_wandb(self, flush_events=False) -> None:
        """Log all events to wandb."""
        if self._wandb.run is not None:
            events_table = self._wandb.Table(
                columns=[
                    "event_type",
                    "event_id",
                    "event_start_time",
                    "event_end_time",
                    "event_total_time",
                ]
            )

            llm_input_output_table = self._wandb.Table(
                columns=[
                    "event_type",
                    "event_id",
                    "input",
                    "output",
                ]
            )

        # Get all event pairs by ID sorted by time.
        event_pairs = self.get_event_pairs()

        for event_pair in event_pairs:
            time_stat = self._get_time_stats(event_pair)

            assert len(event_pair) == 2
            event_pair[0].event_type == event_pair[1].event_type

            events_table.add_data(
                event_pair[0].event_type,
                event_pair[0].id_,
                time_stat.start_time,
                time_stat.end_time,
                time_stat.total_secs,
            )

            if event_pair[0].event_type == CBEventType.LLM:
                event_start = event_pair[0]
                event_end = event_pair[1]

                # Handle event start
                if event_start.payload is not None:
                    payload_keys = list(event_start.payload.keys())
                    assert len(payload_keys) == 2 # Handle two keys: <variable> and "template"

                    payload_keys.remove("template")
                    llm_input_output_table.add_data(
                        event_start.event_type.name,
                        event_start.id_,
                        event_start.payload[payload_keys[0]],
                        str(event_start.payload["template"]),
                    )

                # Handle event end
                if event_end.payload is not None:
                    llm_input_output_table.add_data(
                        event_end.event_type.name,
                        event_end.id_,
                        event_end.payload["response"],
                        event_end.payload["formatted_prompt"],
                    )

        if len(events_table.get_index()) > 0:
            self._wandb.log({"event_table": events_table})

        if len(llm_input_output_table.get_index()) > 0:
            self._wandb.log({"llm_input_output_table": llm_input_output_table})

        if flush_events:
            self.flush_event_logs()

        self._wandb.run.finish()
