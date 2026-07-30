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
    """Map LlamaIndex LLM and embedding events to GenAI invocations."""

    _telemetry: TelemetryHandler = PrivateAttr()
    _active: dict[str, _Invocation] = PrivateAttr(default_factory=dict)

    def __init__(self, telemetry: TelemetryHandler) -> None:
        super().__init__()
        self._telemetry = telemetry

    @classmethod
    def class_name(cls) -> str:
        return "GenAIEventHandler"

    def handle(self, event: BaseEvent, **_: Any) -> None:
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


class LlamaIndexOpenTelemetryGenAI:
    """Register native LlamaIndex GenAI semantic-convention instrumentation."""

    def __init__(
        self,
        tracer_provider: TracerProvider | None = None,
        meter_provider: MeterProvider | None = None,
        logger_provider: LoggerProvider | None = None,
    ) -> None:
        self._telemetry = TelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )
        self._handler = GenAIEventHandler(self._telemetry)

    def start_registering(self) -> None:
        """Start translating LlamaIndex dispatcher events into GenAI telemetry."""
        instrument.get_dispatcher().add_event_handler(self._handler)
