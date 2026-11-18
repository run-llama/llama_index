"""
GenAI semantic convention attributes on LLM spans.

The events below mirror the field shapes of the LLM events in
`llama_index.core.instrumentation.events.llm` (this package does not depend
on llama-index-core), and the `raw` payloads mirror what the OpenAI,
Anthropic and Google GenAI integrations put in `ChatResponse.raw`.
"""

import inspect
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import pytest
from llama_index.observability.otel.base import (
    OTelCompatibleEventHandler,
    OTelCompatibleSpanHandler,
)
from llama_index.observability.otel.utils import usage_to_gen_ai_attributes
from llama_index_instrumentation.base import BaseEvent
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from pydantic import BaseModel


class _Exporter(SpanExporter):
    def __init__(self) -> None:
        self.spans: List[ReadableSpan] = []

    def export(self, spans: Any) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


class _Response(BaseModel):
    raw: Any = None
    additional_kwargs: Dict[str, Any] = {}


class LLMChatStartEvent(BaseEvent):
    model_dict: dict
    additional_kwargs: dict = {}

    @classmethod
    def class_name(cls) -> str:
        return "LLMChatStartEvent"


class LLMChatEndEvent(BaseEvent):
    response: Optional[_Response] = None

    @classmethod
    def class_name(cls) -> str:
        return "LLMChatEndEvent"


class LLMCompletionStartEvent(LLMChatStartEvent):
    @classmethod
    def class_name(cls) -> str:
        return "LLMCompletionStartEvent"


class LLMCompletionEndEvent(LLMChatEndEvent):
    @classmethod
    def class_name(cls) -> str:
        return "LLMCompletionEndEvent"


# What the OpenAI SDK returns (a pydantic model, kept as-is in `raw`).
class _PromptDetails(BaseModel):
    cached_tokens: Optional[int] = None


class _OpenAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Optional[_PromptDetails] = None


class _ChatCompletion(BaseModel):
    id: str = "chatcmpl-1"
    model: str
    usage: Optional[_OpenAIUsage] = None


# Anthropic: `raw=dict(message)`, so `usage` stays an SDK object.
class _AnthropicUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: Optional[int] = None
    cache_creation_input_tokens: Optional[int] = None


def _bound() -> inspect.BoundArguments:
    def fn() -> None: ...

    return inspect.BoundArguments(inspect.signature(fn), OrderedDict())


def _run_llm_span(*events: BaseEvent) -> Dict[str, Any]:
    exporter = _Exporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    handler = OTelCompatibleSpanHandler(tracer=provider.get_tracer("test"))
    event_handler = OTelCompatibleEventHandler(span_handler=handler)
    span_id = "OpenAI.chat-uuid"
    handler.span_enter(id_=span_id, bound_args=_bound())
    for event in events:
        event.span_id = span_id
        event_handler.handle(event)
    handler.span_exit(id_=span_id, bound_args=_bound())
    (span,) = exporter.spans
    # the existing span events are still recorded
    assert [e.name for e in span.events] == [e.class_name() for e in events]
    return dict(span.attributes or {})


def test_openai_chat() -> None:
    raw = _ChatCompletion(
        model="gpt-4o-2024-08-06",
        usage=_OpenAIUsage(
            prompt_tokens=1200,
            completion_tokens=80,
            total_tokens=1280,
            prompt_tokens_details=_PromptDetails(cached_tokens=1024),
        ),
    )
    attrs = _run_llm_span(
        LLMChatStartEvent(model_dict={"class_name": "OpenAI", "model_name": "gpt-4o"}),
        LLMChatEndEvent(response=_Response(raw=raw)),
    )
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.request.model"] == "gpt-4o"
    assert attrs["gen_ai.response.model"] == "gpt-4o-2024-08-06"
    assert attrs["gen_ai.usage.input_tokens"] == 1200
    assert attrs["gen_ai.usage.output_tokens"] == 80
    assert attrs["gen_ai.usage.cache_read.input_tokens"] == 1024


def test_anthropic_cache_tokens_added_to_input() -> None:
    raw = {
        "id": "msg_1",
        "model": "claude-sonnet-4-5",
        "usage": _AnthropicUsage(
            input_tokens=20,
            output_tokens=300,
            cache_read_input_tokens=4000,
            cache_creation_input_tokens=500,
        ),
    }
    attrs = _run_llm_span(
        LLMChatStartEvent(model_dict={"model_name": "claude-sonnet-4-5"}),
        LLMChatEndEvent(response=_Response(raw=raw)),
    )
    # Anthropic's input_tokens excludes cache reads and writes
    assert attrs["gen_ai.usage.input_tokens"] == 4520
    assert attrs["gen_ai.usage.output_tokens"] == 300
    assert attrs["gen_ai.usage.cache_read.input_tokens"] == 4000
    assert attrs["gen_ai.response.model"] == "claude-sonnet-4-5"


def test_anthropic_without_cache() -> None:
    raw = {
        "model": "claude-haiku-4-5",
        "usage": _AnthropicUsage(input_tokens=10, output_tokens=5),
    }
    attrs = _run_llm_span(LLMChatEndEvent(response=_Response(raw=raw)))
    assert attrs["gen_ai.usage.input_tokens"] == 10
    assert attrs["gen_ai.usage.output_tokens"] == 5
    assert "gen_ai.usage.cache_read.input_tokens" not in attrs


def test_gemini_usage_metadata() -> None:
    raw = {
        "content": {"parts": [{"text": "hi"}]},
        "usage_metadata": {
            "prompt_token_count": 900,
            "candidates_token_count": 40,
            "thoughts_token_count": 60,
            "cached_content_token_count": 512,
            "total_token_count": 1000,
        },
    }
    attrs = _run_llm_span(
        LLMChatStartEvent(model_dict={"model": "gemini-2.5-flash"}),
        LLMChatEndEvent(
            response=_Response(
                raw=raw,
                # what the integration also copies here; raw wins
                additional_kwargs={"prompt_tokens": 900, "completion_tokens": 40},
            )
        ),
    )
    assert attrs["gen_ai.request.model"] == "gemini-2.5-flash"
    assert attrs["gen_ai.usage.input_tokens"] == 900
    # thinking tokens are output tokens
    assert attrs["gen_ai.usage.output_tokens"] == 100
    assert attrs["gen_ai.usage.cache_read.input_tokens"] == 512


def test_completion_with_usage_only_in_additional_kwargs() -> None:
    attrs = _run_llm_span(
        LLMCompletionStartEvent(model_dict={"model_name": "gpt-3.5-turbo-instruct"}),
        LLMCompletionEndEvent(
            response=_Response(
                raw=None,
                additional_kwargs={
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                    "total_tokens": 10,
                },
            )
        ),
    )
    assert attrs["gen_ai.operation.name"] == "text_completion"
    assert attrs["gen_ai.usage.input_tokens"] == 7
    assert attrs["gen_ai.usage.output_tokens"] == 3


def test_streamed_anthropic_usage_in_additional_kwargs() -> None:
    # Anthropic streaming puts the running usage in additional_kwargs["usage"]
    attrs = _run_llm_span(
        LLMChatEndEvent(
            response=_Response(
                raw={"type": "message_delta"},
                additional_kwargs={"usage": {"input_tokens": 11, "output_tokens": 22}},
            )
        ),
    )
    assert attrs["gen_ai.usage.input_tokens"] == 11
    assert attrs["gen_ai.usage.output_tokens"] == 22


def test_stream_end_after_span_closed_is_ignored() -> None:
    # stream_chat's span ends when it returns the generator, so the end event
    # can arrive for a span that is already closed: nothing to set, no error.
    exporter = _Exporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    handler = OTelCompatibleSpanHandler(tracer=provider.get_tracer("test"))
    event_handler = OTelCompatibleEventHandler(span_handler=handler)
    handler.span_enter(id_="stream-uuid", bound_args=_bound())
    handler.span_exit(id_="stream-uuid", bound_args=_bound())
    raw = _ChatCompletion(
        model="gpt-4o",
        usage=_OpenAIUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    event_handler.handle(
        LLMChatEndEvent(response=_Response(raw=raw), span_id="stream-uuid")
    )
    (span,) = exporter.spans
    assert "gen_ai.usage.input_tokens" not in (span.attributes or {})
    assert handler._events_by_span == {}


@pytest.mark.parametrize(
    "response",
    [
        None,
        _Response(raw=None),
        _Response(raw={"model": None, "usage": None}),
        _Response(raw=_ChatCompletion(model="gpt-4o", usage=None)),
        _Response(raw="not a dict", additional_kwargs={"prompt_tokens": "7"}),
    ],
)
def test_missing_or_odd_usage_sets_nothing(response: Optional[_Response]) -> None:
    attrs = _run_llm_span(
        LLMChatStartEvent(model_dict={"model_name": "unknown"}),
        LLMChatEndEvent(response=response),
    )
    assert attrs.get("gen_ai.operation.name") == "chat"
    assert not any(k.startswith("gen_ai.usage.") for k in attrs)
    assert "gen_ai.request.model" not in attrs


def test_other_events_get_no_gen_ai_attributes() -> None:
    class SomeOtherEvent(BaseEvent):
        model_dict: dict = {"model": "x"}

        @classmethod
        def class_name(cls) -> str:
            return "SomeOtherEvent"

    attrs = _run_llm_span(SomeOtherEvent())
    assert not any(k.startswith("gen_ai.") for k in attrs)


def test_openai_responses_api_usage_shape() -> None:
    usage = {
        "input_tokens": 300,
        "output_tokens": 50,
        "input_tokens_details": {"cached_tokens": 256},
    }
    assert usage_to_gen_ai_attributes(usage) == {
        "gen_ai.usage.input_tokens": 300,
        "gen_ai.usage.output_tokens": 50,
        "gen_ai.usage.cache_read.input_tokens": 256,
    }
