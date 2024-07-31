from typing import Dict, Any, Optional, List, Type

from .events import Event


class Context:
    def __init__(self, parent: Optional["Context"] = None) -> None:
        # Global state
        if parent:
            self.data = parent.data
        else:
            self.data: Dict[str, Any] = {}

        # Step-specific instance
        self.parent = parent
        self._events_buffer: Dict[Type[Event], Event] = {}

    def collect_events(
        self, ev: Event, expected: List[Type[Event]]
    ) -> Optional[List[Event]]:
        self._events_buffer[type(ev)] = ev

        retval: List[Event] = []
        for e_type in expected:
            e_instance = self._events_buffer.get(e_type)
            if e_instance:
                retval.append(e_instance)

        if len(retval) == len(expected):
            return retval
        return None
