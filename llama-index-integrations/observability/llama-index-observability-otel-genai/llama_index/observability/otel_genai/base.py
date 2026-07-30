"""OpenTelemetry GenAI instrumentation backed by LlamaIndex dispatcher events."""

from __future__ import annotations

from typing import Any, Union

import llama_index_instrumentation as instrument
from llama_index_instrumentation.base.event import BaseEvent
from llama_index_instrumentation.event_handlers import BaseEventHandler
from opentelemetry._logs import LoggerProvider
from opentelemetry.metrics import MeterProvider
from opentelemetry.trace import TracerProvider
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    EmbeddingInvocation,
    InferenceInvocation,
)
from opentelemetry.util.genai.types import InputMessage, OutputMessage, Text
from pydantic import PrivateAttr

from llama_index.core.instrumentation.events.embedding import (
    EmbeddingEndEvent,
    EmbeddingStartEvent,
)
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMChatStartEvent,
    LLMCompletionEndEvent,
    LLMCompletionStartEvent,
)


_Invocation = Union[InferenceInvocation, EmbeddingInvocation]


def _model_value(model_dict: dict[str, Any], *names: str) -> str | None:
    for name in names:
        value = model_dict.get(name)
        if value is not None:
            return str(value)
    return None


def _provider_and_model(model_dict: dict[str, Any]) -> tuple[str, str | None]:
    """Extract portable provider/model values from LlamaIndex model payloads."""
    provider = _model_value(model_dict, "provider", "model_provider")
    model = _model_value(model_dict, "model_name", "model", "model_id")
    return provider or "llama_index", model


def _usage_value(response: Any, *names: str) -> int | None:
    """Read token counts from normalized or provider-specific response metadata."""
    metadata = getattr(response, "additional_kwargs", {}) or {}
    raw = getattr(response, "raw", None)
    if isinstance(raw, dict):
        metadata = {**raw, **metadata}
    usage = metadata.get("usage", metadata)
    if not isinstance(usage, dict):
        return None
    for name in names:
        value = usage.get(name)
        if isinstance(value, int):
            return value
    return None


def _event_key(event: BaseEvent) -> str | None:
    """Use the dispatcher-stamped span id to pair start and end events."""
    return event.span_id


class GenAIEventHandler(BaseEventHandler):
    """
    Map LlamaIndex LLM and embedding events to GenAI invocations.

    The handler consumes dispatcher events emitted by LlamaIndex and maintains
    an invocation until its corresponding end event arrives. It uses
    :class:`opentelemetry.util.genai.handler.TelemetryHandler` to create spans,
    metrics, and optional GenAI detail events that conform to OpenTelemetry's
    GenAI semantic conventions.

    This handler is registered by
    :class:`LlamaIndexOtelGenAIInstrumentor`; users generally should not
    instantiate it directly.

    Args:
        telemetry (TelemetryHandler): The GenAI telemetry handler used to create
            and complete invocations.

    """

    _telemetry: TelemetryHandler = PrivateAttr()
    _active: dict[str, _Invocation] = PrivateAttr(default_factory=dict)

    def __init__(self, telemetry: TelemetryHandler) -> None:
        """
        Initialize the event handler.

        Args:
            telemetry (TelemetryHandler): The configured GenAI telemetry
                handler.

        """
        super().__init__()
        self._telemetry = telemetry

    @classmethod
    def class_name(cls) -> str:
        """
        Return the event handler's stable instrumentation name.

        Returns:
            str: ``"GenAIEventHandler"``.

        """
        return "GenAIEventHandler"

    def handle(self, event: BaseEvent, **_: Any) -> None:
        """
        Translate a supported LlamaIndex event into GenAI telemetry.

        Start events create an invocation and retain it by dispatcher span ID.
        Matching end events enrich and complete that invocation. Events outside
        of a dispatcher span, and event types not currently supported by this
        integration, are ignored.

        Args:
            event (BaseEvent): The LlamaIndex dispatcher event to process.
            **_: Additional dispatcher event-handler arguments, ignored by this
                integration.

        """
        key = _event_key(event)
        if key is None:
            return

        if isinstance(event, LLMChatStartEvent):
            provider, model = _provider_and_model(event.model_dict)
            invocation = self._telemetry.inference(
                provider, request_model=model, operation_name="chat"
            )
            self._set_chat_input(invocation, event.messages)
            self._set_request_parameters(invocation, event.additional_kwargs)
            self._active[key] = invocation
        elif isinstance(event, LLMCompletionStartEvent):
            provider, model = _provider_and_model(event.model_dict)
            invocation = self._telemetry.inference(
                provider, request_model=model, operation_name="completion"
            )
            if self._telemetry.should_capture_content():
                invocation.input_messages = [
                    InputMessage(role="user", parts=[Text(content=event.prompt)])
                ]
            self._set_request_parameters(invocation, event.additional_kwargs)
            self._active[key] = invocation
        elif isinstance(event, EmbeddingStartEvent):
            provider, model = _provider_and_model(event.model_dict)
            self._active[key] = self._telemetry.embedding(provider, request_model=model)
        elif isinstance(event, LLMChatEndEvent):
            invocation = self._pop(key, InferenceInvocation)
            if invocation is not None:
                self._finish_chat(invocation, event)
        elif isinstance(event, LLMCompletionEndEvent):
            invocation = self._pop(key, InferenceInvocation)
            if invocation is not None:
                self._finish_completion(invocation, event)
        elif isinstance(event, EmbeddingEndEvent):
            invocation = self._pop(key, EmbeddingInvocation)
            if invocation is not None:
                if event.embeddings:
                    invocation.dimension_count = len(event.embeddings[0])
                invocation.stop()

    def _pop(self, key: str, expected_type: type[_Invocation]) -> _Invocation | None:
        invocation = self._active.pop(key, None)
        return invocation if isinstance(invocation, expected_type) else None

    def _set_chat_input(
        self, invocation: InferenceInvocation, messages: list[Any]
    ) -> None:
        if not self._telemetry.should_capture_content():
            return
        invocation.input_messages = [
            InputMessage(
                role=str(
                    message.role.value
                    if hasattr(message.role, "value")
                    else message.role
                ),
                parts=[Text(content=message.content or "")],
            )
            for message in messages
        ]

    @staticmethod
    def _set_request_parameters(
        invocation: InferenceInvocation, kwargs: dict[str, Any]
    ) -> None:
        for field in ("temperature", "top_p", "max_tokens", "seed"):
            value = kwargs.get(field)
            if value is not None:
                setattr(invocation, field, value)

    def _finish_chat(
        self, invocation: InferenceInvocation, event: LLMChatEndEvent
    ) -> None:
        response = event.response
        if response is not None:
            if self._telemetry.should_capture_content():
                invocation.output_messages = [
                    OutputMessage(
                        role=str(response.message.role.value),
                        parts=[Text(content=response.message.content or "")],
                        finish_reason=str(
                            response.additional_kwargs.get("finish_reason", "stop")
                        ),
                    )
                ]
            invocation.input_tokens = _usage_value(
                response, "prompt_tokens", "input_tokens"
            )
            invocation.output_tokens = _usage_value(
                response, "completion_tokens", "output_tokens"
            )
            invocation.response_model_name = _model_value(
                response.additional_kwargs, "model", "model_name"
            )
        invocation.stop()

    def _finish_completion(
        self, invocation: InferenceInvocation, event: LLMCompletionEndEvent
    ) -> None:
        response = event.response
        if self._telemetry.should_capture_content():
            invocation.output_messages = [
                OutputMessage(
                    role="assistant",
                    parts=[Text(content=response.text)],
                    finish_reason=str(
                        response.additional_kwargs.get("finish_reason", "stop")
                    ),
                )
            ]
        invocation.input_tokens = _usage_value(
            response, "prompt_tokens", "input_tokens"
        )
        invocation.output_tokens = _usage_value(
            response, "completion_tokens", "output_tokens"
        )
        invocation.response_model_name = _model_value(
            response.additional_kwargs, "model", "model_name"
        )
        invocation.stop()


class LlamaIndexOtelGenAIInstrumentor:
    """
    Register native OpenTelemetry GenAI instrumentation for LlamaIndex.

    This opt-in integration translates LlamaIndex dispatcher events into
    OpenTelemetry GenAI semantic-convention telemetry. It currently supports
    LLM chat/completion and embedding operations. Prompt and response content
    remains disabled unless enabled through the standard
    ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` configuration.

    Args:
        tracer_provider (Optional[TracerProvider]): Provider used to create
            GenAI spans. Uses OpenTelemetry's configured global provider when
            omitted.
        meter_provider (Optional[MeterProvider]): Provider used to record GenAI
            duration and token-usage metrics. Uses the configured global meter
            provider when omitted.
        logger_provider (Optional[LoggerProvider]): Provider used to emit
            optional GenAI detail events. Uses the configured global logger
            provider when omitted.

    """

    def __init__(
        self,
        tracer_provider: TracerProvider | None = None,
        meter_provider: MeterProvider | None = None,
        logger_provider: LoggerProvider | None = None,
    ) -> None:
        """
        Configure native GenAI telemetry for later registration.

        Registration is intentionally separate from construction so callers can
        configure OpenTelemetry providers before LlamaIndex starts emitting
        events.

        Args:
            tracer_provider (Optional[TracerProvider]): Provider for generated
                spans.
            meter_provider (Optional[MeterProvider]): Provider for generated
                metrics.
            logger_provider (Optional[LoggerProvider]): Provider for optional
                GenAI detail events.

        """
        self._telemetry = TelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )
        self._handler = GenAIEventHandler(self._telemetry)

    def start_registering(self) -> None:
        """
        Register the handler with LlamaIndex's global dispatcher.

        After this call, supported LlamaIndex operations emit OpenTelemetry
        GenAI telemetry through the configured providers.
        """
        instrument.get_dispatcher().add_event_handler(self._handler)
