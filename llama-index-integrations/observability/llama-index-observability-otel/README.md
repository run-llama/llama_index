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

# initialize the instrumentation object
instrumentor = LlamaIndexOpenTelemetry()

if __name__ == "__main__":
    # start listening!
    instrumentor.start_registering()
    # register events
    documents = SimpleDirectoryReader(
        input_dir="./data/paul_graham/"
    ).load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    query_result = query_engine.query("Who is Paul?")
    # turn the events into a span and streamline them to OpenTelemetry
    instrumentor.to_otel_traces()
    # register another batch of events
    quere_result_one = query_engine.query("What did Paul do?")
    # turn the events into another span and streamline them to OpenTelemetry
    instrumentor.to_otel_traces()
```

Or you can add some customization to the `LlamaIndexOpenTelemetry` class by, for example, set a custom span exporter or a custom service name:

```python
from llama_index.observability.otel import LlamaIndexOpenTelemetry
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)

# define a custom span exporter
span_exporter = OTLPSpanExporter("http://0.0.0.0:4318/v1/traces")

# initialize the instrumentation object
instrumentor = LlamaIndexOpenTelemetry(
    service_name_or_resource="my.otel.service", span_exporter=span_exporter
)

if __name__ == "__main__":
    # start listening!
    instrumentor.start_registering()
    # register events
    documents = SimpleDirectoryReader(
        input_dir="./data/paul_graham/"
    ).load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    query_result = query_engine.query("Who is Paul?")
    # turn the events into a span and streamline them to OpenTelemetry
    instrumentor.to_otel_traces()
    # register another batch of events
    quere_result_one = query_engine.query("What did Paul do?")
    # turn the events into another span and streamline them to OpenTelemetry
    instrumentor.to_otel_traces()
```
