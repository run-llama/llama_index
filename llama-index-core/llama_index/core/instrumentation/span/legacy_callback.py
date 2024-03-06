from typing import Optional
from llama_index.core.bridge.pydantic import Field
from llama_index.core.instrumentation.span.base import BaseSpan
from llama_index.core.callbacks.base import EventContext


class LegacyCallbackSpan(BaseSpan):
    """Legacy Callback Span class for backwards compatibility."""

    event_context: Optional[EventContext] = Field(
        default=None, description="Associated EventContext."
    )
