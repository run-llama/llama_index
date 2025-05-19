import inspect
from termcolor.termcolor import cprint
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.bridge.pydantic import BaseModel, Field, ConfigDict, PrivateAttr
from datetime import datetime
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.span_handlers.simple import SimpleSpanHandler
import llama_index.core.instrumentation as instrument
from llama_index.core.instrumentation.span import SimpleSpan
from typing import Optional, Any, List, Dict, Union, Sequence, Literal, Mapping
from llama_index.observability.otel.utils import filter_model_fields
from opentelemetry import trace, context
from opentelemetry.sdk.trace import TracerProvider, _Span
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    ConsoleSpanExporter
)


class OTelEventAttributes(BaseModel):
    name: str
    attributes: Optional[Mapping[str, Union[str, bool, int, float, Sequence[str], Sequence[bool], Sequence[int], Sequence[float]]]]


class OTelCompatibleSpanHandler(SimpleSpanHandler):
    """OpenTelemetry-compatible span handler."""

    _tracer: trace.Tracer = PrivateAttr()
    all_spans: Dict[str, Union[trace.Span, _Span]] = Field(
        default_factory=dict,
        description="All the registered OpenTelemetry spans."
    )
    all_events: List[OTelEventAttributes] = Field(
        default_factory=list,
        description="All the registered OpenTelemetry events."
    )
    debug: bool = Field(
        default=False,
        description="Debug the start and end of span and the recording of events"
    )

    def __init__(
        self,
        tracer: trace.Tracer,
        all_spans: Dict[str, Union[trace.Span, _Span]] = {},
        all_events: List[OTelEventAttributes] = [],
        debug: bool = False,
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
        self._tracer = tracer
        self.all_spans = all_spans
        self.all_events = all_events
        self.debug = debug

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
        ctx = context.Context(bound_args.arguments)
        otel_span = self._tracer.start_span(name=id_, context=ctx)
        self.all_spans.update({id_: otel_span})
        if self.debug:
            cprint(f"Emitting span {id_} at time: {datetime.now()}", color="yellow", attrs=["bold"])
        return span

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any,
    ) -> SimpleSpan:
        if self.debug:
            cprint(f"Preparing to end span {id_} at time: {datetime.now()}", color="blue", attrs=["bold"])
        sp = super().prepare_to_exit_span(id_, bound_args, instance, result, **kwargs)
        span = self.all_spans[id_]
        for event in self.all_events:
            span.add_event(name=event.name, attributes=event.attributes)
        self.all_events.clear()
        span.set_status(status=trace.StatusCode.OK)
        span.end()
        return sp

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        err: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> Optional[SimpleSpan]:
        if self.debug:
            cprint(f"Preparing to exit span {id_} with an error at time: {datetime.now()}", color="red", attrs=["bold"])
        sp = super().prepare_to_drop_span(id_, bound_args, instance, err, **kwargs)
        span = self.all_spans[id_]
        for event in self.all_events:
            span.add_event(name=event.name, attributes=event.attributes)
        self.all_events.clear()
        span.set_status(status=trace.StatusCode.ERROR, description=err.__str__())
        span.end()
        return sp

class OTelCompatibleEventHandler(BaseEventHandler):
    """OpenTelemetry-compatible event handler."""

    span_handler: OTelCompatibleSpanHandler = Field(
        description="Span Handler associated with the event handler"
    )
    debug: bool = Field(
        default=False,
        description="Debug the start and end of span and the recording of events"
    )

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "OtelCompatibleEventHandler"

    def handle(self, event: BaseEvent, **kwargs: Any) -> None:
        """Handle events by pushing them in a list that will serve for forwarding the events to OpenTelemetry."""
        if self.debug:
            cprint(f"Registering a {event.class_name()} event at time: {datetime.now()}", color="green", attrs=["bold"])
        otel_event = OTelEventAttributes(name=event.class_name(), attributes=filter_model_fields(event.dict()))
        self.span_handler.all_events.append(otel_event)

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
    debug: bool = Field(
        default=False,
        description="Debug the start and end of span and the recording of events"
    )
    _tracer: Optional[trace.Tracer] = PrivateAttr(
        default=None
    )

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
        self._tracer = trace.get_tracer("llamaindex.opentelemetry.tracer")

    def start_registering(
        self,
    ) -> None:
        """Starts LlamaIndex instrumentation."""
        self._start_otel()
        dispatcher = instrument.get_dispatcher(self.dispatcher_name)
        span_handler = OTelCompatibleSpanHandler(tracer=self._tracer, debug=self.debug)
        dispatcher.add_span_handler(span_handler)
        dispatcher.add_event_handler(OTelCompatibleEventHandler(span_handler=span_handler, debug=self.debug))
