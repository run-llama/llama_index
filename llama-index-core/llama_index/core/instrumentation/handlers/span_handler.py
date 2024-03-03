from typing import Any, Dict, Generic, Optional, TypeVar
from abc import abstractmethod
from llama_index.core.bridge.pydantic import Field
from llama_index.core.instrumentation.span.base import BaseSpan
from llama_index.core.instrumentation.handlers.base import BaseEventHandler
from llama_index.core.instrumentation.events.base import BaseEvent


T = TypeVar("T", bound=BaseSpan)


class BaseSpanHandler(BaseEventHandler, Generic[T]):
    spans: Dict[str, BaseSpan] = Field(
        default_factory=Dict, description="Dictionary of spans."
    )
    current_span_id: Optional[str] = Field(
        default=None, description="Id of current span."
    )

    def class_name(cls) -> str:
        """Class name."""
        return "SpanHandler"

    def handle(self, event: BaseEvent) -> None:
        """Handle logic - null handler does nothing."""
        return

    def span_enter(self, id: str) -> None:
        """Logic for entering a span."""
        if id in self.spans:
            pass
        else:
            # TODO: thread safe
            span = self.new_span()
            span.parent_id = self.current_span_id
            self.spans[id] = span

    def span_exit(self, id: str) -> None:
        """Logic for exiting a span."""
        self.drop_span(id)

    @abstractmethod
    def new_span(self) -> T:
        """Create a span."""
        ...

    @abstractmethod
    def drop_span(self, id: str) -> Any:
        """Logic for droppping a span."""
        ...
