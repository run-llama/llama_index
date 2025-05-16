import inspect
import uuid
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.bridge.pydantic import BaseModel, Field, ConfigDict, PrivateAttr
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.span_handlers.simple import SimpleSpanHandler
import llama_index.core.instrumentation as instrument
from llama_index.core.instrumentation.span import SimpleSpan
from typing import Optional, Any, List, Dict, Union, Sequence, Literal, Mapping
from utils import filter_model_fields
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    ConsoleSpanExporter
)

class OTelCompatibleSpanHandler(SimpleSpanHandler):
    """OpenTelemetry-compatible span handler."""

    all_spans: List[SimpleSpan] = Field(
        default_factory=list,
        description="List to temporarily collect all spans."
    )
    def __init__(
        self,
        all_spans: List[SimpleSpan] = [],
        open_spans: Dict[str, SimpleSpan] = {},
        completed_spans: List[SimpleSpan] = [],
        dropped_spans: List[SimpleSpan] = [],
        current_span_ids: Dict[Any, str] = {},
    ):
        super().__init__(
            open_spans=open_spans,
            completed_spans=completed_spans,
            dropped_spans=dropped_spans,
            current_span_ids=current_span_ids,
        )
        self.all_spans = all_spans

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "OTelCompatibleSpanHandler"

    def new_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> SimpleSpan:
        span = super().new_span(id_, bound_args, instance, parent_span_id, tags, **kwargs)
        self.all_spans.append(span)
        return span

class OTelEventAttributes(BaseModel):
    name: str
    attributes: Optional[Mapping[str, Union[str, bool, int, float, Sequence[str], Sequence[bool], Sequence[int], Sequence[float]]]]

class OTelCompatibleEventHandler(BaseEventHandler):
    """OpenTelemetry-compatible event handler."""

    all_events: List[OTelEventAttributes] = Field(
        default_factory=list,
        description="A list of all events for tracing purposes"
    )

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "OtelCompatibleEventHandler"

    def handle(self, event: BaseEvent, **kwargs: Any) -> None:
        """Handle events by pushing them in a list that will serve for forwarding the events to OpenTelemetry."""
        self.all_events.append(OTelEventAttributes(name=event.class_name(), attributes=filter_model_fields(event.dict())))

class LlamaIndexOpenTelemetry(BaseModel):
    """
    LlamaIndexOpenTelemetry is a configuration and integration class for OpenTelemetry tracing within LlamaIndex.
    This class manages the setup and registration of OpenTelemetry span and event handlers, configures the tracer provider,
    and exports trace data using the specified span exporter and processor. It supports both simple and batch span processors,
    and allows customization of the service name or resource, as well as the dispatcher name.

    Attributes:
        span_exporter (Optional[SpanExporter]): The OpenTelemetry span exporter. Defaults to ConsoleSpanExporter.
        span_processor (Literal["simple", "batch"]): The span processor type, either 'simple' or 'batch'. Defaults to 'batch'.
        service_name_or_resource (Union[str, Resource]): The service name or OpenTelemetry Resource. Defaults to a Resource with service name 'llamaindex.opentelemetry'.
        dispatcher_name (str): The name for the LlamaIndex dispatcher. Defaults to 'root'.

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    span_exporter: Optional[SpanExporter] = Field(
        default=ConsoleSpanExporter(),
        description="OpenTelemetry span exporter. Supports all SpanExporter-compatible interfaces, defaults to ConsoleSpanExporter."
    )
    span_processor: Literal["simple", "batch"] = Field(
        default="batch",
        description="OpenTelemetry span processor. Can be either 'batch' (-> BatchSpanProcessor) or 'simple' (-> SimpleSpanProcessor). Defaults to 'batch'"
    )
    service_name_or_resource: Union[str, Resource] = Field(
        default=Resource(attributes={SERVICE_NAME: "llamaindex.opentelemetry"}),
        description="Service name or resource for OpenTelemetry. Defaults to a Resource with 'llamaindex.opentelemetry' as service name."
    )
    dispatcher_name: str = Field(
        default="root",
        description="Name for LlamaIndex dispatcher. Defaults to 'root'"
    )
    _span_handler: OTelCompatibleSpanHandler = PrivateAttr(
        default=OTelCompatibleSpanHandler()
    )
    _event_handler: OTelCompatibleEventHandler = PrivateAttr(
        default=OTelCompatibleEventHandler()
    )
    _is_otel_started: bool = PrivateAttr(
        default=False
    )

    def start_registering(
        self,
    ) -> None:
        """Starts LlamaIndex instrumentation."""
        dispatcher = instrument.get_dispatcher(self.dispatcher_name)
        dispatcher.add_event_handler(self._event_handler)
        dispatcher.add_span_handler(self._span_handler)

    def _start_otel(
        self,
    ) -> None:
        if isinstance(self.service_name_or_resource, str):
            self.service_name_or_resource = Resource(attributes={SERVICE_NAME: self.service_name_or_resource})
        tracer_provider = TracerProvider(resource=self.service_name_or_resource)
        if self.span_processor == "simple":
            span_processor = SimpleSpanProcessor(self.span_exporter)
        else:
            span_processor = BatchSpanProcessor(self.span_exporter)
        tracer_provider.add_span_processor(span_processor=span_processor)
        trace.set_tracer_provider(tracer_provider)
        self._is_otel_started = True

    def to_otel_traces(
        self,
    ) -> None:
        """Converts LlamaIndex instrumentation outputs into OpenTelemetry-compatible spans and events."""
        if not self._is_otel_started:
            self._start_otel()
        events = self._event_handler.all_events
        tracer = trace.get_tracer(f"llamaindex.opentelemetry.tracer")
        with tracer.start_as_current_span(str(uuid.uuid4())) as span:
            for event in events:
                span.add_event(name=event.name, attributes=event.attributes)
        self._event_handler.all_events.clear()
