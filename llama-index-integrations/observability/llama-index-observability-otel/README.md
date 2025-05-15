# LlamaIndex OpenTelemetry Observability Integration

## Installation

```shell
pip install llama-index-observability-otel
```

## Usage

You can use the default OpenTelemetry event handler as follows:

```python
from llama_index.observability.otel import OpenTelemetryEventHandler
import llama_index.core.instrumentation as instrument
from llama_index.core.instrumentation.span_handlers import SimpleSpanHandler
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# initialize the EventHandler
event_handler = OpenTelemetryEventHandler()
# initialize all the other components needed for instrumentation
span_handler = SimpleSpanHandler()
dispatcher = instrument.get_dispatcher()
dispatcher.add_event_handler(event_handler)
dispatcher.add_span_handler(span_handler)

if __name__ == "__main__":
    # try it out with a simple RAG example!
    documents = SimpleDirectoryReader(
        input_dir="./data/paul_graham/"
    ).load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    query_result = query_engine.query("Who is Paul?")
```

Or you can add some customization by instantiating a `TracerOperator` object

```python
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from llama_index.observability.otel import (
    OpenTelemetryEventHandler,
    TracerOperator,
)
import llama_index.core.instrumentation as instrument
from llama_index.core.instrumentation.span_handlers import SimpleSpanHandler
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# define a custom span exporter
span_exporter = OTLPSpanExporter("0.0.0.0:4318/v1/traces")

# define some custom OpenTelemetry components
tracer_operator = TracerOperator(
    tracer_name="my.test.project",
    span_exporter=span_exporter,
    span_processor="simple",
)

# initialize the EventHandler
event_handler = OpenTelemetryEventHandler(tracer_operator=tracer_operator)

# initialize all the other components needed for instrumentation
span_handler = SimpleSpanHandler()
dispatcher = instrument.get_dispatcher()
dispatcher.add_event_handler(event_handler)
dispatcher.add_span_handler(span_handler)


if __name__ == "__main__":
    # try it out with a simple RAG example!
    documents = SimpleDirectoryReader(
        input_dir="./data/paul_graham/"
    ).load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    query_result = query_engine.query("Who is Paul?")
```
