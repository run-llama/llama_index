from llama_index.observability.otel import LlamaIndexOpenTelemetry
from llama_index.observability.otel.base import (
    Resource,
    SERVICE_NAME,
    ConsoleSpanExporter,
)
from llama_index.observability.otel.utils import flatten_dict


def test_initialization() -> None:
    instrumentor = LlamaIndexOpenTelemetry()
    assert instrumentor.service_name_or_resource == Resource(
        attributes={SERVICE_NAME: "llamaindex.opentelemetry"}
    )
    assert isinstance(instrumentor.span_exporter, ConsoleSpanExporter)
    assert instrumentor.span_processor == "batch"
    assert instrumentor._tracer is None
    assert not instrumentor.debug


def test_diff_initialization() -> None:
    instrumentor = LlamaIndexOpenTelemetry(
        service_name_or_resource="this.is.a.test",
        span_processor="simple",
        debug=True,
    )
    assert instrumentor.service_name_or_resource == "this.is.a.test"
    assert isinstance(instrumentor.span_exporter, ConsoleSpanExporter)
    assert instrumentor.span_processor == "simple"
    assert instrumentor._tracer is None
    assert instrumentor.debug


def test_flatten_dict() -> None:
    nested_dict = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    flattened = flatten_dict(nested_dict)
    assert flattened == {"a": 1, "b.c": 2, "b.d.e": 3}

    nested_dict = {"a": 1, "b": {"c": 2, "d": {"e": 3}}, "f": [1, 2, 3]}
    flattened = flatten_dict(nested_dict)
    assert flattened == {"a": 1, "b.c": 2, "b.d.e": 3, "f": [1, 2, 3]}
