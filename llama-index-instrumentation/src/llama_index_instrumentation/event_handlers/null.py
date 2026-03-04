from typing import Any

from llama_index_instrumentation.base import BaseEvent

from .base import BaseEventHandler


class NullEventHandler(BaseEventHandler):
    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "NullEventHandler"

    def handle(self, event: BaseEvent, **kwargs: Any) -> Any:
        """Handle logic - null handler does nothing."""
        return
