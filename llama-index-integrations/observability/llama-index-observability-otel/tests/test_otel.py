import functools
import inspect
from collections import OrderedDict
from typing import Any, Optional, Sequence

from llama_index.observability.otel import LlamaIndexOpenTelemetry
from llama_index.observability.otel.base import (
    SERVICE_NAME,
    ConsoleSpanExporter,
    OTelCompatibleSpanHandler,
    Resource,
)
from llama_index.observability.otel.utils import flatten_dict
from opentelemetry import context, trace
from opentelemetry.sdk.trace import Event, ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.util.instrumentation import (
    InstrumentationInfo,
    InstrumentationScope,
)
from opentelemetry.trace.span import SpanContext
from opentelemetry.trace.status import Status
from opentelemetry.util.types import Attributes


def test_initialization() -> None:
    instrumentor = LlamaIndexOpenTelemetry()
    assert instrumentor.service_name_or_resource == Resource(
        attributes={SERVICE_NAME: "llamaindex.opentelemetry"}
    )
    assert instrumentor.tracer_provider is None
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
    assert instrumentor.tracer_provider is None
    assert instrumentor.service_name_or_resource == "this.is.a.test"
    assert isinstance(instrumentor.span_exporter, ConsoleSpanExporter)
    assert instrumentor.span_processor == "simple"
    assert instrumentor._tracer is None
    assert instrumentor.debug


def test_initialize_with_extra_span_processors() -> None:
    class CustomSpanProcessor(SpanProcessor):
        pass

    instrumentor = LlamaIndexOpenTelemetry(
        service_name_or_resource="this.is.a.test",
        span_processor="simple",
        debug=True,
        extra_span_processors=[CustomSpanProcessor()],
    )
    assert instrumentor.tracer_provider is None
    assert instrumentor.service_name_or_resource == "this.is.a.test"
    assert isinstance(instrumentor.span_exporter, ConsoleSpanExporter)
    assert instrumentor.span_processor == "simple"
    assert len(instrumentor.extra_span_processors) == 1
    assert all(
        isinstance(span_processor, CustomSpanProcessor)
        for span_processor in instrumentor.extra_span_processors
    )
    assert instrumentor._tracer is None
    assert instrumentor.debug


def test_init_with_custom_tracer_provider() -> None:
    tracer_provider = TracerProvider(
        resource=Resource(attributes={}, schema_url="test.schema.url")
    )
    instrumentor = LlamaIndexOpenTelemetry(
        service_name_or_resource="this.is.a.test",
        span_processor="simple",
        debug=True,
        tracer_provider=tracer_provider,
    )
    assert instrumentor.tracer_provider is not None
    assert instrumentor.tracer_provider.resource is not None
    assert instrumentor.tracer_provider.resource.schema_url == "test.schema.url"


class MockContext(dict):
    pass


class MockSpan(ReadableSpan):
    def __init__(
        self,
        name: str,
        context: Optional[SpanContext] = None,
        parent: Optional[SpanContext] = None,
        resource: Optional[Resource] = None,
        attributes: Attributes = None,
        events: Sequence[Event] = (),
        links: Sequence[trace.Link] = (),
        kind: trace.SpanKind = trace.SpanKind.INTERNAL,
        instrumentation_info: Optional[InstrumentationInfo] = None,
        status: Status = Status(),
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        instrumentation_scope: Optional[InstrumentationScope] = None,
        dict_context: Optional[context.Context] = None,
    ) -> None:
        super().__init__(
            name,
            context,
            parent,
            resource,
            attributes,
            events,
            links,
            kind,
            instrumentation_info,
            status,
            start_time,
            end_time,
            instrumentation_scope,
        )
        self.dict_context = dict_context


class MockTracer(trace.Tracer):
    def __init__(self) -> None:
        super().__init__()

    def start_span(
        self,
        name: str,
        context: Optional[context.Context] = None,
        kind: trace.SpanKind = trace.SpanKind.INTERNAL,
        attributes: Optional[Attributes] = None,
        links: Any = None,
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> Any:
        return MockSpan(
            name,
            kind=kind,
            attributes=attributes,
            links=links,
            start_time=start_time,
            dict_context=context,
        )

    def start_as_current_span(self, *args, **kwargs) -> Any:
        raise NotImplementedError("not implemented")


@functools.lru_cache(maxsize=1)
def get_ordered_dict() -> OrderedDict:
    d = OrderedDict()
    d["arg1"] = "hello"
    d["arg2"] = 1
    return d


def test_content_inheritance() -> None:
    span_handler = OTelCompatibleSpanHandler(
        tracer=MockTracer(),
        debug=False,
    )

    context.attach(context.Context({"hello": "world"}))

    assert context.get_current().get("hello") == "world"

    def instrumented_fn(arg1: str, arg2: int) -> str:
        return arg1 + str(arg2)

    span_handler.new_span(
        id_="1",
        bound_args=inspect.BoundArguments(
            signature=inspect.signature(instrumented_fn),
            arguments=get_ordered_dict(),
        ),
    )

    assert len(span_handler.all_spans) == 1
    span = span_handler.all_spans.get("1")
    assert isinstance(span, MockSpan)
    assert span.dict_context is not None
    assert span.dict_context.get("hello") == "world"
    # ensure preservation of the bound arguments
    assert span.dict_context.get("arg1") == "hello"
    assert span.dict_context.get("arg2") == 1


def test_context_inheritance_empty_context() -> None:
    span_handler = OTelCompatibleSpanHandler(
        tracer=MockTracer(),
        debug=False,
    )

    context.attach(context.Context())  # attach empty context

    assert len(list(context.get_current().keys())) == 0

    def instrumented_fn(arg1: str, arg2: int) -> str:
        return arg1 + str(arg2)

    span_handler.new_span(
        id_="1",
        bound_args=inspect.BoundArguments(
            signature=inspect.signature(instrumented_fn),
            arguments=get_ordered_dict(),
        ),
    )

    assert len(span_handler.all_spans) == 1
    span = span_handler.all_spans.get("1")
    assert isinstance(span, MockSpan)
    assert span.dict_context is not None
    assert span.dict_context.get("hello") is None
    # ensure preservation of the bound arguments
    assert span.dict_context.get("arg1") == "hello"
    assert span.dict_context.get("arg2") == 1


def test_flatten_dict() -> None:
    nested_dict = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    flattened = flatten_dict(nested_dict)
    assert flattened == {"a": 1, "b.c": 2, "b.d.e": 3}

    nested_dict = {"a": 1, "b": {"c": 2, "d": {"e": 3}}, "f": [1, 2, 3]}
    flattened = flatten_dict(nested_dict)
    assert flattened == {"a": 1, "b.c": 2, "b.d.e": 3, "f": [1, 2, 3]}
