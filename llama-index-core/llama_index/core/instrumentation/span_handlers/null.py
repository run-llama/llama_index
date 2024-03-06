from typing import Optional
from llama_index.core.instrumentation.span_handlers.base import BaseSpanHandler
from llama_index.core.instrumentation.span.base import BaseSpan


class NullSpanHandler(BaseSpanHandler[BaseSpan]):
    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "NullSpanHandler"

    def span_enter(self, id: str) -> None:
        """Logic for entering a span."""
        return

    def span_exit(self, id: str) -> None:
        """Logic for exiting a span."""
        return

    def new_span(self) -> None:
        """Create a span."""
        return

    def prepare_to_exit_span(self, id: str, **kwargs) -> None:
        """Logic for exiting a span."""
        return

    def prepare_to_drop_span(self, id: str, err: Optional[Exception], **kwargs) -> None:
        """Logic for droppping a span."""
        return
