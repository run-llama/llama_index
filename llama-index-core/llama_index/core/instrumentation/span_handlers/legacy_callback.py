from typing import Any, Optional
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks.base import EventContext, EventPayload, CBEventType
from llama_index.core.instrumentation.span import LegacyCallbackSpan
from llama_index.core.instrumentation.span_handlers.base import BaseSpanHandler


class LegacyCallbackSpanHandler(BaseSpanHandler[LegacyCallbackSpan]):
    """Our Legacy Callback Manager as a SpanHandler for backwards compatibility."""

    callback_manager: CallbackManager = Field(
        default_factory=CallbackManager,
        description="Callback manager to manage spans.",
    )

    def class_name(cls) -> str:
        """Class name."""
        return "LegacyCallbackSpanHandler"

    def new_span(
        self, id: str, parent_span_id: Optional[str], **kwargs: Any
    ) -> LegacyCallbackSpan:
        """Create a span."""
        if "event_type" in kwargs:  # EventContext type of span
            event_context = EventContext(self.callback_manager, **kwargs)
            event_context.on_start(payload=kwargs["payload"])
        else:  # trace type of span
            self.callback_manager.start_trace(id)
            event_context = None
        return LegacyCallbackSpan(
            id_=id, parent_id=parent_span_id, event_context=event_context
        )

    def prepare_to_exit_span(self, id: str, **kwargs: Any) -> None:
        """Logic for preparing to drop a span."""
        event_context = self.open_spans[id].event_context
        if event_context and not event_context.finished:
            event_context.on_end()
        self.callback_manager.end_trace(id)

    def prepare_to_drop_span(
        self, id: str, err: Optional[Exception], **kwargs: Any
    ) -> None:
        """Logic for preparing to drop a span."""
        event_context = self.open_spans[id].event_context

        if not hasattr(err, "event_added"):
            payload = {EventPayload.EXCEPTION: err}
            if not event_context:
                # trace error handling
                self.callback_manager.on_event_start(
                    CBEventType.EXCEPTION, payload=payload
                )
                err.event_added = True  # type: ignore
                self.callback_manager.end_trace(id)
            else:
                # event context error handling
                err.event_added = True  # type: ignore
                if not event_context.finished:
                    event_context.on_end(payload=payload)
