from typing import Any
from llama_index.core.event_management.handlers.base import BaseEventHandler


class NullHandler(BaseEventHandler):
    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "NullHandler"

    def handle(self, event: Any) -> None:
        """Handle logic - null handler does nothing."""
        return

    def span_enter(self, id: str) -> None:
        """Logic for entering a span."""
        return

    def span_exit(self, id: str) -> None:
        """Logic for exiting a span."""
        return
