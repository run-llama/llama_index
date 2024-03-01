from typing import Any
from llama_index.core.event_management.handlers.base import BaseEventHandler


class NullHandler(BaseEventHandler):
    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "NullHandler"

    def handle(event: Any) -> None:
        """Handle logic - null handler does nothing."""
