import uuid
from collections.abc import Sequence as ABCSequence
from typing import Optional, Literal, Any
from typing_extensions import Self
from pydantic import BaseModel, model_validator
from opentelemetry import trace
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    ConsoleSpanExporter
)

class TracerOperator(BaseModel):
    """
    TracerOperator is a Pydantic model that encapsulates the configuration and initialization of OpenTelemetry tracing components for observability in the LlamaIndex integration.

    Attributes
    ----------
        `tracer_name (Optional[str])`: The name to use for the tracer.
        `span_exporter (Optional[SpanExporter])`: The exporter responsible for sending spans to a backend or console.
        `span_processor (Optional[Literal['batch', 'simple']])`: The processor that manages span exporting.
        `tracer (Optional[Tracer])`: The tracer instance used to create spans.

    Config
    ------
        `arbitrary_types_allowed (bool)`: Allows arbitrary types for model fields.

    Methods
    -------
        `validate_exporter(self) -> Self`:
            Pydantic model validator that ensures all tracing components are properly initialized.
            If not provided, it sets up default exporter, processor, and provider, and registers the tracer.

    """

    tracer_name: Optional[str] = None
    span_exporter: Optional[SpanExporter] = None
    span_processor: Optional[Literal["batch", "simple"]] = None
    tracer: Optional[trace.Tracer] = None

    class Config:
        arbitrary_types_allowed=True

    @model_validator(mode="after")
    def validate_exporter(self) -> Self:
        if self.tracer:
            return self
        if not self.tracer_name:
            self.tracer_name = "llamaindex.opentelemetry"
        if not self.span_exporter:
            self.span_exporter = ConsoleSpanExporter()
        if not self.span_processor or self.span_processor == "batch":
            span_processor = BatchSpanProcessor(self.span_exporter)
        elif self.span_processor == "simple":
            span_processor = SimpleSpanProcessor(self.span_exporter)

        resource = Resource(attributes={SERVICE_NAME: self.tracer_name})
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(span_processor)
        trace.set_tracer_provider(tracer_provider)

        self.tracer = trace.get_tracer("llamaindex.opentelemetry.tracer")

        return self

base_types = (int, str, bool, bytes, float)

def _is_otel_supported_type(obj: Any) -> bool:
    # If it's one of the base types
    if isinstance(obj, base_types):
        return True

    # If it's a sequence (but not a string or bytes, which are sequences too)
    if isinstance(obj, ABCSequence) and not isinstance(obj, (str, bytes)):
        return all(isinstance(item, base_types) for item in obj)

    return False

def _filter_model_fields(model_dict: dict) -> dict:
    newdct =  {}
    for field in model_dict:
        if _is_otel_supported_type(model_dict[field]):
            newdct.update({field: model_dict[field]})

    return newdct

class OpenTelemetryEventHandler(BaseEventHandler):
    """
    A simple event handler for OpenTelemetry, that listens and traces every event happening within a LlamaIndex environment.

    Attributes
    ----------
        `tracer_operator (Optional[TracerOperator])`: The `TracerOperator` object that initializes OpenTelemetry instrumentation, containing a tracer or a tracer provider, a span processor and a span exporter. Defaults to the default values of the `TracerOperator` class.

    Methods
    -------
        `class_name() -> str`:
            Returns the name of the class
        `handle(event: BaseEvent, **kwargs: Any) -> None`:
            Captures events and exports them within spans.

    """

    tracer_operator: Optional[TracerOperator] = None

    @model_validator(mode="after")
    def validate_tracer(self) -> Self:
        if not self.tracer_operator:
            self.tracer_operator = TracerOperator()

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "OpenTelemetryEventHandler"

    def handle(self, event: BaseEvent, **kwargs: Any) -> None:
        with self.tracer_operator.tracer.start_as_current_span(name=str(uuid.uuid4())) as span:
            span.add_event(name=event.class_name(), attributes=_filter_model_fields(event.model_dump()))
