from typing import cast, List, Optional
from llama_index.core.bridge.pydantic import Field
from llama_index.core.instrumentation.span.simple import SimpleSpan
from llama_index.core.instrumentation.span_handlers.base import BaseSpanHandler
from datetime import datetime


class SimpleSpanHandler(BaseSpanHandler[SimpleSpan]):
    """Span Handler that managest SimpleSpan's."""

    completed_spans: List[SimpleSpan] = Field(
        default_factory=list, description="List of completed spans."
    )

    def class_name(cls) -> str:
        """Class name."""
        return "SimpleSpanHandler"

    def new_span(self, id: str, parent_span_id: Optional[str]) -> SimpleSpan:
        """Create a span."""
        return SimpleSpan(id_=id, parent_id=parent_span_id)

    def prepare_to_exit_span(self, id: str, **kwargs) -> None:
        """Logic for preparing to drop a span."""
        span = self.open_spans[id]
        span = cast(SimpleSpan, span)
        span.end_time = datetime.now()
        span.duration = (span.end_time - span.start_time).total_seconds()
        self.completed_spans += [span]

    def prepare_to_drop_span(self, id: str, err: Optional[Exception], **kwargs) -> None:
        """Logic for droppping a span."""
        return
