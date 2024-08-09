from collections import defaultdict
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
        self._events_buffer: Dict[Type[Event], List[Event]] = defaultdict(list)

    def collect_events(
        self, ev: Event, expected: List[Type[Event]]
    ) -> Optional[List[Event]]:
        self._events_buffer[type(ev)].append(ev)

        retval: List[Event] = []
        for e_type in expected:
            e_instance_list = self._events_buffer.get(e_type)
            if e_instance_list:
                retval.append(e_instance_list.pop(0))

        if len(retval) == len(expected):
            return retval

        # put back the events if unable to collect all
        for ev in retval:
            self._events_buffer[type(ev)].append(ev)

        return None
