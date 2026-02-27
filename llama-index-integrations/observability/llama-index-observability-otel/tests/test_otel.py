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


## ---------------------------------------------------------------------------
# Integration tests using InMemorySpanExporter
# ---------------------------------------------------------------------------

from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor as _SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)


class InMemorySpanExporter(SpanExporter):
    """Minimal in-memory exporter for testing."""

    def __init__(self):
        self._spans = []

    def export(self, spans):
        self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def get_finished_spans(self):
        return list(self._spans)

    def shutdown(self):
        pass


def _fn(arg1: str = "hello") -> str:
    return arg1


_bound = inspect.BoundArguments(
    signature=inspect.signature(_fn),
    arguments=OrderedDict({"arg1": "hello"}),
)


def make_handler():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(_SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    handler = OTelCompatibleSpanHandler(tracer=tracer)
    return exporter, handler, provider


def test_span_name_strips_uuid() -> None:
    exporter, handler, provider = make_handler()
    handler.span_enter(id_="MyWorkflow.run-abc123-def", bound_args=_bound)
    handler.span_exit(id_="MyWorkflow.run-abc123-def", bound_args=_bound)
    provider.force_flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "MyWorkflow.run"


def test_missing_parent_no_crash() -> None:
    exporter, handler, provider = make_handler()
    handler.span_enter(
        id_="span1-uuid",
        bound_args=_bound,
        parent_id="nonexistent-123",
    )
    handler.span_exit(id_="span1-uuid", bound_args=_bound)
    provider.force_flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1


def test_error_records_exception() -> None:
    exporter, handler, provider = make_handler()
    handler.span_enter(id_="err-span-uuid", bound_args=_bound)
    err = ValueError("boom")
    handler.span_drop(id_="err-span-uuid", bound_args=_bound, err=err)
    provider.force_flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    events = spans[0].events
    exception_events = [e for e in events if e.name == "exception"]
    assert len(exception_events) >= 1
    assert exception_events[0].attributes["exception.message"] == "boom"


def test_tags_recorded_as_attributes() -> None:
    exporter, handler, provider = make_handler()
    handler.span_enter(
        id_="tag-span-uuid",
        bound_args=_bound,
        tags={
            "handler_id": "h1",
            "run_id": "r1",
            "myapp.custom": "user_val",
            "parent_span_id": "skip_me",
            "_otel_traceparent": "skip_too",
        },
    )
    handler.span_exit(id_="tag-span-uuid", bound_args=_bound)
    provider.force_flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes)
    assert attrs["llamaindex.handler_id"] == "h1"
    assert attrs["llamaindex.run_id"] == "r1"
    # Dotted keys pass through without prefix
    assert attrs["myapp.custom"] == "user_val"
    assert "llamaindex.parent_span_id" not in attrs
    assert "llamaindex._otel_traceparent" not in attrs


def test_non_otel_types_skipped_in_tags() -> None:
    exporter, handler, provider = make_handler()
    handler.span_enter(
        id_="type-span-uuid",
        bound_args=_bound,
        tags={
            "good": "yes",
            "bad_dict": {"nested": True},
            "bad_none": None,
        },
    )
    handler.span_exit(id_="type-span-uuid", bound_args=_bound)
    provider.force_flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes)
    assert attrs["llamaindex.good"] == "yes"
    assert "llamaindex.bad_dict" not in attrs
    assert "llamaindex.bad_none" not in attrs


def test_traceparent_injected_on_root_span() -> None:
    exporter, handler, provider = make_handler()
    tags: dict = {}
    handler.span_enter(id_="root-span-uuid", bound_args=_bound, tags=tags)
    assert "_otel_traceparent" in tags
    assert tags["_otel_traceparent"].startswith("00-")
    handler.span_exit(id_="root-span-uuid", bound_args=_bound)
    provider.force_flush()


def test_traceparent_restored_on_recovery() -> None:
    # Create first handler + root span to get a traceparent
    exporter1, handler1, provider1 = make_handler()
    tags: dict = {}
    handler1.span_enter(id_="root-uuid", bound_args=_bound, tags=tags)
    captured_tp = tags["_otel_traceparent"]
    handler1.span_exit(id_="root-uuid", bound_args=_bound)
    provider1.force_flush()
    root_trace_id = exporter1.get_finished_spans()[0].context.trace_id

    # New handler — simulates recovery in a different process/context
    exporter2, handler2, provider2 = make_handler()
    handler2.span_enter(
        id_="child-uuid",
        bound_args=_bound,
        parent_id="gone-span",
        tags={"_otel_traceparent": captured_tp},
    )
    handler2.span_exit(id_="child-uuid", bound_args=_bound)
    provider2.force_flush()
    child_spans = exporter2.get_finished_spans()
    assert len(child_spans) == 1
    # The child span should belong to the same trace as the root
    assert child_spans[0].context.trace_id == root_trace_id


def test_flatten_dict() -> None:
    nested_dict = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    flattened = flatten_dict(nested_dict)
    assert flattened == {"a": 1, "b.c": 2, "b.d.e": 3}

    nested_dict = {"a": 1, "b": {"c": 2, "d": {"e": 3}}, "f": [1, 2, 3]}
    flattened = flatten_dict(nested_dict)
    assert flattened == {"a": 1, "b.c": 2, "b.d.e": 3, "f": [1, 2, 3]}
