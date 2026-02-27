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


def test_tags_not_mutated_by_new_span() -> None:
    """Regression guard: new_span must not write internal keys into the tags dict."""
    exporter, handler, provider = make_handler()
    tags: dict = {"user_key": "user_val"}
    original_tags = dict(tags)
    handler.span_enter(id_="root-span-uuid", bound_args=_bound, tags=tags)
    handler.span_exit(id_="root-span-uuid", bound_args=_bound)
    provider.force_flush()
    # Tags should be unchanged — no _otel_traceparent or other internal keys injected
    assert tags == original_tags


def test_capture_propagation_context() -> None:
    """capture_propagation_context returns a dict with traceparent when a span is active."""
    exporter, handler, provider = make_handler()
    # Create a span so there's an active trace context
    handler.span_enter(id_="root-uuid", bound_args=_bound)
    # Activate the OTel span in the current context so capture can see it
    from opentelemetry.trace import set_span_in_context

    otel_span = handler.all_spans["root-uuid"]
    ctx = set_span_in_context(otel_span)
    context.attach(ctx)

    captured = handler.capture_propagation_context()
    assert "otel" in captured
    assert captured["otel"]["traceparent"].startswith("00-")

    handler.span_exit(id_="root-uuid", bound_args=_bound)
    provider.force_flush()


def test_capture_restore_propagation_roundtrip() -> None:
    """
    Full roundtrip: capture context in process A, restore in process B.

    Verifies:
    - trace_id continuity (child belongs to same trace as parent)
    - parent_span_id linkage (child's parent points to root span)
    - tags become span attributes on both sides independently
    - tags dict is not mutated by either side
    - externally-set OTel context (e.g. baggage-like ambient values) propagates
      through the traceparent mechanism
    """
    from opentelemetry.trace import set_span_in_context

    # --- Process A: create root span with tags, capture context ---
    exporter_a, handler_a, provider_a = make_handler()

    tags_a = {"handler_id": "h1", "run_id": "r1", "myapp.custom": "val_a"}
    original_tags_a = dict(tags_a)
    handler_a.span_enter(id_="root-uuid", bound_args=_bound, tags=tags_a)

    # Activate the OTel span in ambient context (simulating what the Dispatcher
    # would do before a serialization boundary)
    root_otel_span = handler_a.all_spans["root-uuid"]
    context.attach(set_span_in_context(root_otel_span))

    # Capture propagation context — this is what gets serialized across the boundary
    captured_ctx = handler_a.capture_propagation_context()
    assert "otel" in captured_ctx
    assert captured_ctx["otel"]["traceparent"].startswith("00-")

    # Tags were NOT mutated by capture or span creation
    assert tags_a == original_tags_a

    # Finish root span
    handler_a.span_exit(id_="root-uuid", bound_args=_bound)
    provider_a.force_flush()
    root_spans = exporter_a.get_finished_spans()
    assert len(root_spans) == 1
    root_span = root_spans[0]
    root_trace_id = root_span.context.trace_id
    root_span_id = root_span.context.span_id

    # Root span has tags as attributes
    root_attrs = dict(root_span.attributes)
    assert root_attrs["llamaindex.handler_id"] == "h1"
    assert root_attrs["llamaindex.run_id"] == "r1"
    assert root_attrs["myapp.custom"] == "val_a"

    # --- Process B: restore context, create child span with its own tags ---
    exporter_b, handler_b, provider_b = make_handler()

    # Clean ambient context (simulating a fresh process)
    context.attach(context.Context())

    # Restore the captured propagation context
    handler_b.restore_propagation_context(captured_ctx)

    # Create child span — parent_id references a span not in this handler's all_spans,
    # so it falls back to ambient OTel context (which we just restored)
    tags_b = {"handler_id": "h2", "run_id": "r2"}
    original_tags_b = dict(tags_b)
    handler_b.span_enter(
        id_="child-uuid",
        bound_args=_bound,
        parent_id="gone-span",
        tags=tags_b,
    )
    handler_b.span_exit(id_="child-uuid", bound_args=_bound)
    provider_b.force_flush()

    child_spans = exporter_b.get_finished_spans()
    assert len(child_spans) == 1
    child_span = child_spans[0]

    # Trace continuity: same trace_id
    assert child_span.context.trace_id == root_trace_id

    # Parent linkage: child's parent is the root span
    assert child_span.parent is not None
    assert child_span.parent.span_id == root_span_id
    assert child_span.parent.trace_id == root_trace_id

    # Child has its own tags as attributes (independent of process A's tags)
    child_attrs = dict(child_span.attributes)
    assert child_attrs["llamaindex.handler_id"] == "h2"
    assert child_attrs["llamaindex.run_id"] == "r2"
    # Process A's tags are NOT on the child span
    assert "myapp.custom" not in child_attrs

    # Tags were NOT mutated by span creation on either side
    assert tags_b == original_tags_b


def test_dispatcher_propagation_roundtrip_with_tags() -> None:
    """
    Full Dispatcher-level roundtrip: trace context + instrument_tags propagate together.

    Simulates: process A runs a workflow step with run_id tag,
    captures context, process B restores it and creates a child span
    that inherits the trace AND sees the tags.
    """
    from llama_index_instrumentation.dispatcher import (
        Dispatcher,
        Manager,
        active_instrument_tags,
        instrument_tags,
    )
    from opentelemetry.trace import set_span_in_context

    exporter_a, handler_a, provider_a = make_handler()
    exporter_b, handler_b, provider_b = make_handler()

    # Set up two dispatchers (simulating two processes, each with their own handler)
    dispatcher_a = Dispatcher(
        name="process_a", span_handlers=[handler_a], propagate=False
    )
    dispatcher_a.manager = Manager(dispatcher_a)

    dispatcher_b = Dispatcher(
        name="process_b", span_handlers=[handler_b], propagate=False
    )
    dispatcher_b.manager = Manager(dispatcher_b)

    # --- Process A: run with tags, capture context ---
    with instrument_tags({"run_id": "run-123", "handler_id": "wf-abc"}):
        dispatcher_a.span_enter(
            id_="root-uuid", bound_args=_bound, tags=active_instrument_tags.get()
        )

        # Activate OTel span in ambient context
        root_otel_span = handler_a.all_spans["root-uuid"]
        context.attach(set_span_in_context(root_otel_span))

        captured = dispatcher_a.capture_propagation_context()

    # Verify captured structure has both namespaces
    assert "otel" in captured
    assert "instrument_tags" in captured
    assert captured["instrument_tags"]["run_id"] == "run-123"
    assert captured["instrument_tags"]["handler_id"] == "wf-abc"

    dispatcher_a.span_exit(id_="root-uuid", bound_args=_bound)
    provider_a.force_flush()
    root_span = exporter_a.get_finished_spans()[0]

    # --- Process B: restore context, create child ---
    context.attach(context.Context())  # clean slate
    dispatcher_b.restore_propagation_context(captured)

    # instrument_tags should now be active
    restored_tags = active_instrument_tags.get()
    assert restored_tags["run_id"] == "run-123"
    assert restored_tags["handler_id"] == "wf-abc"

    dispatcher_b.span_enter(
        id_="child-uuid",
        bound_args=_bound,
        parent_id="gone-span",
        tags=restored_tags,
    )
    dispatcher_b.span_exit(id_="child-uuid", bound_args=_bound)
    provider_b.force_flush()

    child_span = exporter_b.get_finished_spans()[0]

    # Trace continuity
    assert child_span.context.trace_id == root_span.context.trace_id
    assert child_span.parent.span_id == root_span.context.span_id

    # Tags became attributes on the child span
    child_attrs = dict(child_span.attributes)
    assert child_attrs["llamaindex.run_id"] == "run-123"
    assert child_attrs["llamaindex.handler_id"] == "wf-abc"


def test_flatten_dict() -> None:
    nested_dict = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    flattened = flatten_dict(nested_dict)
    assert flattened == {"a": 1, "b.c": 2, "b.d.e": 3}

    nested_dict = {"a": 1, "b": {"c": 2, "d": {"e": 3}}, "f": [1, 2, 3]}
    flattened = flatten_dict(nested_dict)
    assert flattened == {"a": 1, "b.c": 2, "b.d.e": 3, "f": [1, 2, 3]}
