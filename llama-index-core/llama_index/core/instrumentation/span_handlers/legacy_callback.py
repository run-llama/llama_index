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
        event_context = None
        if "query_bundle" in kwargs:  # EventContext type of span
            query_bundle = kwargs["query_bundle"]
            if "._query" in id:
                payload = {EventPayload.QUERY_STR: query_bundle.query_str}
                event_context = EventContext(
                    self.callback_manager,
                    CBEventType.QUERY,
                )
                event_context.on_start(payload=payload)
        elif "message" in kwargs:  # stream_chat
            message = kwargs["message"]
            if "._achat" in id:
                payload = {EventPayload.MESSAGES: [message]}
                event_context = EventContext(
                    self.callback_manager, CBEventType.AGENT_STEP
                )
                event_context.on_start(payload=payload)
        elif "tool_call" in kwargs:  # tool call
            if "call_function" in id and "._call_function" not in id:
                tool_call = kwargs["tool_call"]
                function_call = tool_call.function
                tools = kwargs["tools"]
                tool = next(
                    tool for tool in tools if tool.metadata.name == function_call.name
                )
                payload = {
                    EventPayload.FUNCTION_CALL: function_call.arguments,
                    EventPayload.TOOL: tool.metadata,
                }
                event_context = EventContext(
                    self.callback_manager, CBEventType.FUNCTION_CALL
                )
                event_context.on_start(payload=payload)

        else:  # trace type of span
            self.callback_manager.start_trace(id)
        return LegacyCallbackSpan(
            id_=id, parent_id=parent_span_id, event_context=event_context
        )

    def prepare_to_exit_span(
        self, id: str, result: Optional[Any] = None, **kwargs: Any
    ) -> None:
        """Logic for preparing to drop a span."""
        # defining here to avoid circular imports
        from llama_index.core.chat_engine.types import StreamingAgentChatResponse
        from llama_index.core.base.response.schema import RESPONSE_TYPE

        event_context = self.open_spans[id].event_context
        if isinstance(result, RESPONSE_TYPE):
            payload = {EventPayload.RESPONSE: result}
        elif isinstance(result, StreamingAgentChatResponse):
            payload = {EventPayload.RESPONSE: result}
        else:
            payload = {}
        if event_context and not event_context.finished:
            event_context.on_end(payload=payload)
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
