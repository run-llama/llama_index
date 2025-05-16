import inspect
from collections.abc import Sequence as ABCSequence
from typing import Optional, Literal, Any, List, Dict, cast
from typing_extensions import Self
from pydantic import BaseModel, model_validator
from opentelemetry import trace
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.span_handlers import BaseSpanHandler
from llama_index.core.instrumentation.span import BaseSpan
from opentelemetry.sdk.trace import TracerProvider, Span
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    ConsoleSpanExporter
)

class OpenTelemetrySpan(BaseSpan):
    otel_span: Span

class TracerOperator(BaseModel):
    """
    TracerOperator is a Pydantic model that encapsulates the configuration and initialization of OpenTelemetry tracing components for observability in the LlamaIndex integration.

    Attributes:
        tracer_name (Optional[str]): The name to use for the tracer.
        span_exporter (Optional[SpanExporter]): The exporter responsible for sending spans to a backend or console.
        span_processor (Optional[Literal['batch', 'simple']]): The processor that manages span exporting.
        tracer (Optional[Tracer]): The tracer instance used to create spans.

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

class OpenTelemetrySpanHandler(BaseSpanHandler):
    tracer_operator: Optional[TracerOperator] = None
    open_span_ids: List[str] = []
    @model_validator(mode="after")
    def validate_span_handler(self) -> Self:
        if not self.tracer_operator:
            self.tracer_operator = TracerOperator()
        return self
    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "OpenTelemetrySpanHandler"

    def new_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[List[Dict[str, Any]]] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[OpenTelemetrySpan]:
        """Create a span."""
        # get parent context
        if parent_span_id is not None:
            parent_span_context = self.open_spans[parent_span_id].otel_span.get_span_context()
            link_from_parent_span = [trace.Link(parent_span_context)]
            # move parent span to completed spans
            self.completed_spans.append(self.open_spans[parent_span_id])
            self.open_spans.pop(parent_span_id)
        else:
            link_from_parent_span = None
        otel_span = self.tracer_operator.tracer.start_span(name=id_, links=link_from_parent_span)
        self.open_span_ids.append(id_)
        self.open_spans.update({id_: OpenTelemetrySpan(id_=id_,
        parent_id=parent_span_id, tags=tags,otel_span=otel_span)})
        return OpenTelemetrySpan(id_=id_, parent_id=parent_span_id, tags=tags,otel_span=otel_span)

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any,
    ) -> OpenTelemetrySpan:
        """Logic for preparing to drop a span."""
        span = self.open_spans[id_]
        span = cast(OpenTelemetrySpan, span)
        span.otel_span.end()
        with self.lock:
            self.completed_spans += [span]
        return span

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        err: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> Optional[OpenTelemetrySpan]:
        """Logic for droppping a span."""
        if id_ in self.open_spans:
            with self.lock:
                span = self.open_spans[id_]
                span.otel_span.set_status(trace.Status(trace.StatusCode.ERROR))
                self.dropped_spans += [span]
            return span
        return None

class OpenTelemetryEventHandler(BaseEventHandler):
    """
    A simple event handler for OpenTelemetry, that listens and traces every event happening within a LlamaIndex environment.

    Attributes:
        tracer_operator (Optional[TracerOperator]): The `TracerOperator` object that initializes OpenTelemetry instrumentation, containing a tracer or a tracer provider, a span processor and a span exporter. Defaults to the default values of the `TracerOperator` class.

    """

    span_handler: OpenTelemetrySpanHandler

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "OpenTelemetryEventHandler"

    def handle(self, event: BaseEvent, **kwargs: Any) -> None:
        span_id = self.span_handler.open_span_ids[-1]
        otel_span = self.span_handler.open_spans[span_id].otel_span
        with otel_span as span:
            span.add_event(name=event.class_name(), attributes=_filter_model_fields(event.dict()))
