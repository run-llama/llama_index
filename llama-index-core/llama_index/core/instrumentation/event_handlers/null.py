from typing import Any
from llama_index.core.instrumentation.event_handlers.base import BaseEventHandler
from llama_index.core.instrumentation.events.base import BaseEvent


class NullEventHandler(BaseEventHandler):
    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "NullEventHandler"

    def handle(self, event: BaseEvent, **kwargs) -> Any:
        """Handle logic - null handler does nothing."""
        return
