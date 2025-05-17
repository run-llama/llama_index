from llama_index.observability.otel import LlamaIndexOpenTelemetry
from llama_index.observability.otel.base import Resource, SERVICE_NAME, ConsoleSpanExporter, OTelCompatibleSpanHandler, OTelCompatibleEventHandler

def test_initialization() -> None:
    instrumentor = LlamaIndexOpenTelemetry()
    assert instrumentor.service_name_or_resource == Resource(attributes={SERVICE_NAME: "llamaindex.opentelemetry"})
    assert isinstance(instrumentor.span_exporter, ConsoleSpanExporter)
    assert instrumentor.span_processor == "batch"
    assert isinstance(instrumentor._span_handler, OTelCompatibleSpanHandler)
    assert isinstance(instrumentor._span_handler.all_spans, list)
    assert isinstance(instrumentor._event_handler, OTelCompatibleEventHandler)
    assert isinstance(instrumentor._event_handler.all_events, list)
    assert not instrumentor._is_otel_started

def test_diff_initialization() -> None:
    instrumentor = LlamaIndexOpenTelemetry(
        service_name_or_resource="this.is.a.test",
        span_processor="simple"
    )
    assert instrumentor.service_name_or_resource == "this.is.a.test"
    assert isinstance(instrumentor.span_exporter, ConsoleSpanExporter)
    assert instrumentor.span_processor == "simple"
    assert isinstance(instrumentor._span_handler, OTelCompatibleSpanHandler)
    assert isinstance(instrumentor._span_handler.all_spans, list)
    assert isinstance(instrumentor._event_handler, OTelCompatibleEventHandler)
    assert isinstance(instrumentor._event_handler.all_events, list)
    assert not instrumentor._is_otel_started
