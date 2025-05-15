from llama_index.observability.otel import OpenTelemetryEventHandler, TracerOperator
from llama_index.observability.otel.base import SpanExporter, trace

def test_initialization() -> None:
    event_handler = OpenTelemetryEventHandler()
    assert event_handler.tracer_operator.tracer_name == "llamaindex.opentelemetry"
    assert isinstance(event_handler.tracer_operator.span_exporter, SpanExporter)
    assert isinstance(event_handler.tracer_operator.tracer, trace.Tracer)

def test_diff_initialization() -> None:
    tracer_operator = TracerOperator(
        tracer_name = "my.test.project",
    )
    event_handler = OpenTelemetryEventHandler(tracer_operator=tracer_operator)
    assert event_handler.tracer_operator.tracer_name == "my.test.project"
    assert isinstance(event_handler.tracer_operator.span_exporter, SpanExporter)
    assert isinstance(event_handler.tracer_operator.tracer, trace.Tracer)
