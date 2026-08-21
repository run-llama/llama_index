# LlamaIndex OpenTelemetry Observability Integration

## Installation

```shell
pip install llama-index-observability-otel
```

## Usage

You can use the default OpenTelemetry observability class as follows:

```python
from llama_index.observability.otel import LlamaIndexOpenTelemetry
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.llms import MockLLM
from llama_index.core.embeddings import MockEmbedding
from llama_index.core import Settings

# initialize the instrumentation object
instrumentor = LlamaIndexOpenTelemetry()

if __name__ == "__main__":
    embed_model = MockEmbedding(embed_dim=256)
    llm = MockLLM()
    Settings.embed_model = embed_model
    # start listening!
    instrumentor.start_registering()
    # register events
    documents = SimpleDirectoryReader(
        input_dir="./data/paul_graham/"
    ).load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(llm=llm)
    query_result = query_engine.query("Who is Paul?")
    query_result_one = query_engine.query("What did Paul do?")
```

Or you can add some customization to the `LlamaIndexOpenTelemetry` class by, for example, set a custom span exporter, a custom service name, activating the debugging, set a custom list of extra span processors...

```python
from llama_index.observability.otel import LlamaIndexOpenTelemetry
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk.trace import SpanProcessor
from llama_index.core.llms import MockLLM
from llama_index.core.embeddings import MockEmbedding
from llama_index.core import Settings


class CustomSpanProcessor(SpanProcessor):
    # your implementation
    ...


# define a custom span exporter
span_exporter = OTLPSpanExporter("http://0.0.0.0:4318/v1/traces")

# initialize the instrumentation object
instrumentor = LlamaIndexOpenTelemetry(
    service_name_or_resource="my.test.service.1",
    span_exporter=span_exporter,
    debug=True,
    extra_span_processors=[CustomSpanProcessor()],
)

if __name__ == "__main__":
    embed_model = MockEmbedding(embed_dim=256)
    llm = MockLLM()
    Settings.embed_model = embed_model
    # start listening!
    instrumentor.start_registering()
    # register events
    documents = SimpleDirectoryReader(
        input_dir="./data/paul_graham/"
    ).load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(llm=llm)
    query_result = query_engine.query("Who is Paul?")
    query_result_one = query_engine.query("What did Paul do?")
```

## GenAI semantic conventions

Enable GenAI telemetry to emit standard OpenTelemetry GenAI spans and metrics
for LlamaIndex LLM, embedding, retrieval, tool, and agent operations:

```python
from llama_index.observability.otel import LlamaIndexOpenTelemetry

instrumentor = LlamaIndexOpenTelemetry(genai_enabled=True)
instrumentor.start_registering()
```

Prompt and response content is disabled by default. Configure
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY` to opt in to
content capture on spans. The accepted values are `NO_CONTENT`, `SPAN_ONLY`,
`EVENT_ONLY`, and `SPAN_AND_EVENT`.

### Provider instrumentation

If an application also enables OpenTelemetry instrumentation for its underlying
model provider, both integrations can emit generation spans for the same
request. To avoid duplicate generation telemetry, enable one source of LLM
generation spans, or use both intentionally when the provider-specific detail
is desired.
