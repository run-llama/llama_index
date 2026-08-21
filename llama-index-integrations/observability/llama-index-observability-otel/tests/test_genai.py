from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult

from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.chat_engine.types import AgentChatResponse
from llama_index.core.instrumentation.events.agent import (
    AgentChatWithStepEndEvent,
    AgentChatWithStepStartEvent,
    AgentToolCallEvent,
    AgentToolCallEndEvent,
    AgentToolCallStartEvent,
)
from llama_index.core.instrumentation.events.embedding import (
    EmbeddingEndEvent,
    EmbeddingStartEvent,
)
from llama_index.core.instrumentation.events.llm import (
    LLMCompletionEndEvent,
    LLMCompletionStartEvent,
)
from llama_index.core.instrumentation.events.retrieval import (
    RetrievalEndEvent,
    RetrievalStartEvent,
)
from llama_index.observability.otel.genai import GenAIEventHandler
from llama_index.observability.otel import LlamaIndexOpenTelemetry
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.tools.types import ToolMetadata, ToolOutput


class InMemoryExporter:
    def __init__(self) -> None:
        self.spans = []

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


def _handler() -> tuple[GenAIEventHandler, InMemoryExporter]:
    exporter = InMemoryExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return GenAIEventHandler(tracer_provider=provider), exporter


def test_genai_can_be_enabled_on_public_integration() -> None:
    assert LlamaIndexOpenTelemetry(genai_enabled=True).genai_enabled


def test_completion_records_genai_model_and_usage() -> None:
    handler, exporter = _handler()
    handler.handle(
        LLMCompletionStartEvent(
            span_id="completion",
            prompt="hello",
            additional_kwargs={"temperature": 0.2},
            model_dict={"provider": "openai", "model_name": "gpt-test"},
        )
    )
    handler.handle(
        LLMCompletionEndEvent(
            span_id="completion",
            prompt="hello",
            response=CompletionResponse(
                text="hi",
                additional_kwargs={
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2}
                },
            ),
        )
    )

    span = exporter.spans[0]
    assert span.attributes["gen_ai.operation.name"] == "completion"
    assert span.attributes["gen_ai.provider.name"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-test"
    assert span.attributes["gen_ai.usage.input_tokens"] == 3
    assert span.attributes["gen_ai.usage.output_tokens"] == 2


def test_embedding_records_dimension() -> None:
    handler, exporter = _handler()
    handler.handle(
        EmbeddingStartEvent(
            span_id="embedding",
            model_dict={"provider": "openai", "model_name": "text-embedding-3-small"},
        )
    )
    handler.handle(
        EmbeddingEndEvent(
            span_id="embedding", chunks=["hello"], embeddings=[[0.1, 0.2, 0.3]]
        )
    )

    span = exporter.spans[0]
    assert span.attributes["gen_ai.operation.name"] == "embeddings"
    assert span.attributes["gen_ai.embeddings.dimension.count"] == 3


def test_retrieval_records_span() -> None:
    handler, exporter = _handler()
    handler.handle(
        RetrievalStartEvent(span_id="retrieval", str_or_query_bundle="hello")
    )
    handler.handle(
        RetrievalEndEvent(
            span_id="retrieval",
            str_or_query_bundle="hello",
            nodes=[NodeWithScore(node=TextNode(text="world"), score=0.8)],
        )
    )

    assert exporter.spans[0].attributes["gen_ai.operation.name"] == "retrieval"


def test_tool_call_records_span() -> None:
    handler, exporter = _handler()
    handler.handle(
        AgentToolCallEvent(
            arguments='{"city": "New York"}',
            tool=ToolMetadata(name="weather", description="Get the weather"),
        )
    )

    span = exporter.spans[0]
    assert span.attributes["gen_ai.operation.name"] == "execute_tool"
    assert span.attributes["gen_ai.tool.name"] == "weather"


def test_tool_lifecycle_records_duration_and_result() -> None:
    handler, exporter = _handler()
    handler.handle(
        AgentToolCallStartEvent(
            tool_id="call-1",
            tool_name="weather",
            tool_kwargs={"city": "New York"},
            tool_description="Get the weather",
        )
    )
    handler.handle(
        AgentToolCallEndEvent(
            tool_id="call-1",
            tool_name="weather",
            tool_output=ToolOutput(
                content="Sunny",
                tool_name="weather",
                raw_input={"city": "New York"},
                raw_output="Sunny",
            ),
        )
    )

    span = exporter.spans[0]
    assert span.attributes["gen_ai.operation.name"] == "execute_tool"
    assert span.attributes["gen_ai.tool.call.id"] == "call-1"
    assert span.attributes["gen_ai.tool.name"] == "weather"


def test_agent_records_span() -> None:
    handler, exporter = _handler()
    handler.handle(AgentChatWithStepStartEvent(span_id="agent", user_msg="hello"))
    handler.handle(
        AgentChatWithStepEndEvent(
            span_id="agent", response=AgentChatResponse(response="hi")
        )
    )

    assert exporter.spans[0].attributes["gen_ai.operation.name"] == "invoke_agent"
