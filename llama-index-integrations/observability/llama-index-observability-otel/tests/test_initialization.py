from llama_index.observability.otel import OpenTelemetryEventHandler, OpenTelemetrySpanHandler, TracerOperator
from llama_index.observability.otel.base import SpanExporter, trace, BaseEventHandler

def test_initialization() -> None:
    span_handler = OpenTelemetrySpanHandler()
    assert span_handler.tracer_operator.tracer_name == "llamaindex.opentelemetry"
    assert isinstance(span_handler.tracer_operator.span_exporter, SpanExporter)
    assert isinstance(span_handler.tracer_operator.tracer, trace.Tracer)

def test_diff_initialization() -> None:
    tracer_operator = TracerOperator(
        tracer_name = "my.test.project",
    )
    span_handler = OpenTelemetrySpanHandler(tracer_operator=tracer_operator)
    assert span_handler.tracer_operator.tracer_name == "my.test.project"
    assert isinstance(span_handler.tracer_operator.span_exporter, SpanExporter)
    assert isinstance(span_handler.tracer_operator.tracer, trace.Tracer)

def test_event_handler_initialization() -> None:
    span_handler = OpenTelemetrySpanHandler()
    event_handler = OpenTelemetryEventHandler(span_handler=span_handler)
    assert isinstance(event_handler, BaseEventHandler)
