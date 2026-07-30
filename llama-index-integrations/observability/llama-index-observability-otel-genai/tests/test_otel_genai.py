from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.instrumentation.events.embedding import (
    EmbeddingEndEvent,
    EmbeddingStartEvent,
)
from llama_index.core.instrumentation.events.llm import (
    LLMCompletionEndEvent,
    LLMCompletionStartEvent,
)
from llama_index.observability.otel_genai import LlamaIndexOtelGenAIInstrumentor
from llama_index.observability.otel_genai.base import GenAIEventHandler
from opentelemetry.util.genai.handler import TelemetryHandler


class InMemoryExporter:
    def __init__(self):
        self.spans = []

    def export(self, spans):
        self.spans.extend(spans)
        from opentelemetry.sdk.trace.export import SpanExportResult

        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


def _handler():
    exporter = InMemoryExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return GenAIEventHandler(TelemetryHandler(tracer_provider=provider)), exporter


def test_public_instrumentor_name():
    assert LlamaIndexOtelGenAIInstrumentor.__name__ == "LlamaIndexOtelGenAIInstrumentor"


def test_completion_records_genai_model_and_usage():
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


def test_embedding_records_dimension():
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
