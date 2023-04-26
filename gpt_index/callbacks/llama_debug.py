from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

from gpt_index.callbacks.base import BaseCallbackHandler
from gpt_index.callbacks.schema import CBEvent, CBEventType


class LlamaDebugHandler(BaseCallbackHandler):
    """Callback handler that keeps track of debug info.

    This handler simply keeps track of event starts/ends, separated by event types. 
    You can use this callback handler to keep track of and debug events.
    """

    def __init__(self, event_starts_to_ignore: List[CBEventType], event_ends_to_ignore: List[CBEventType]) -> None:
        """Initialize the llama debug handler."""
        self.events: Dict[CBEventType, List[CBEvent]] = defaultdict(list)
        self.sequential_events: List[CBEvent] = []
        super().__init__(event_starts_to_ignore=event_starts_to_ignore, event_ends_to_ignore=event_ends_to_ignore)
    
    def on_event_start(self, event_type: CBEventType, event: CBEvent, **kwargs: Any) -> None:
        """Store event start data by event type."""
        self.events[event_type].append(event)
        self.sequential_events.append(event)

    def on_event_end(self, event_type: CBEventType, event: CBEvent, **kwargs: Any) -> None:
        """Store event end data by event type."""
        self.events[event_type].append(event)
        self.sequential_events.append(event)

    def get_events(self, event_type: Optional[CBEventType] = None) -> List[CBEvent]:
        """Get all events for a specific event type."""
        if event_type is not None:
            return self.events[event_type]

        return self.sequential_events

    def _get_paired_events(self, events: List[CBEvent]) -> Dict[str, List[CBEvent, CBEvent]]:
        """Helper function to pair events according to their ID."""
        event_pairs: Dict[str, List[CBEvent, CBEvent]] = defaultdict(list)
        for event in events:
            event_pairs[event.id].append(event)
        return event_pairs

    def get_event_pairs(self, event_type: CBEventType):
        """Pair events by ID, either all events or a sepcific type."""
        if event_type is not None:
            return self._get_paired_events(self.events[event_type])
        
        return self._get_paired_events(self.sequential_events)
