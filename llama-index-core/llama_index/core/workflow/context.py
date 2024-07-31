from typing import Dict, Any, Optional

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
        self._params_buffer: Dict[str, Any] = {}

    def collect_params(self, ev: Event, *params) -> Optional[Dict[str, Any]]:
        for param in params:
            if hasattr(ev, param):
                self._params_buffer[param] = getattr(ev, param)

        if list(self._params_buffer.keys()) == list(params):
            return self._params_buffer
        return None
