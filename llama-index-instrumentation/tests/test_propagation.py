import inspect
from typing import Any, Dict

from llama_index_instrumentation.dispatcher import (
    Dispatcher,
    Manager,
    active_instrument_tags,
)
from llama_index_instrumentation.span_handlers.simple import SimpleSpanHandler


def _make_bound_args():
    return inspect.signature(lambda: None).bind()


class PropagatingHandler(SimpleSpanHandler):
    """Handler that captures/restores a fake trace context."""

    def capture_propagation_context(self) -> Dict[str, Any]:
        return {"test_handler": {"trace_id": "abc123", "span_id": "def456"}}

    def restore_propagation_context(self, context: Dict[str, Any]) -> None:
        self._restored_context = context


def test_capture_propagation_context_basic():
    handler = PropagatingHandler()
    d = Dispatcher(span_handlers=[handler], propagate=False)

    ctx = d.capture_propagation_context()

    assert ctx["test_handler"]["trace_id"] == "abc123"
    assert ctx["test_handler"]["span_id"] == "def456"


def test_capture_includes_instrument_tags():
    handler = SimpleSpanHandler()
    d = Dispatcher(span_handlers=[handler], propagate=False)

    token = active_instrument_tags.set({"user_id": "u1", "session": "s1"})
    try:
        ctx = d.capture_propagation_context()
    finally:
        active_instrument_tags.reset(token)

    assert ctx["instrument_tags"] == {"user_id": "u1", "session": "s1"}


def test_capture_omits_instrument_tags_when_empty():
    handler = SimpleSpanHandler()
    d = Dispatcher(span_handlers=[handler], propagate=False)

    ctx = d.capture_propagation_context()

    assert "instrument_tags" not in ctx


def test_restore_propagation_context_basic():
    handler = PropagatingHandler()
    d = Dispatcher(span_handlers=[handler], propagate=False)

    context = {"test_handler": {"trace_id": "abc123"}}
    d.restore_propagation_context(context)

    assert handler._restored_context == context


def test_restore_sets_instrument_tags():
    handler = SimpleSpanHandler()
    d = Dispatcher(span_handlers=[handler], propagate=False)

    d.restore_propagation_context({"instrument_tags": {"user_id": "u1"}})

    assert active_instrument_tags.get() == {"user_id": "u1"}
    # cleanup
    active_instrument_tags.set({})


def test_capture_walks_parent_chain():
    parent_handler = PropagatingHandler()
    child_handler = SimpleSpanHandler()

    parent = Dispatcher(name="parent", span_handlers=[parent_handler], propagate=False)
    child = Dispatcher(
        name="child",
        span_handlers=[child_handler],
        propagate=True,
        parent_name="parent",
    )
    manager = Manager(parent)
    manager.add_dispatcher(child)
    child.manager = manager
    parent.manager = manager

    ctx = child.capture_propagation_context()

    # Should include parent handler's context via propagation
    assert "test_handler" in ctx


def test_restore_walks_parent_chain():
    parent_handler = PropagatingHandler()
    child_handler = PropagatingHandler()

    parent = Dispatcher(name="parent", span_handlers=[parent_handler], propagate=False)
    child = Dispatcher(
        name="child",
        span_handlers=[child_handler],
        propagate=True,
        parent_name="parent",
    )
    manager = Manager(parent)
    manager.add_dispatcher(child)
    child.manager = manager
    parent.manager = manager

    context = {"test_handler": {"trace_id": "xyz"}}
    child.restore_propagation_context(context)

    assert child_handler._restored_context == context
    assert parent_handler._restored_context == context


def test_capture_stops_at_propagate_false():
    parent_handler = PropagatingHandler()
    child_handler = SimpleSpanHandler()

    parent = Dispatcher(name="parent", span_handlers=[parent_handler], propagate=False)
    child = Dispatcher(
        name="child",
        span_handlers=[child_handler],
        propagate=False,  # does NOT propagate
        parent_name="parent",
    )
    manager = Manager(parent)
    manager.add_dispatcher(child)
    child.manager = manager
    parent.manager = manager

    ctx = child.capture_propagation_context()

    # Should NOT include parent handler's context
    assert "test_handler" not in ctx


def test_roundtrip_capture_restore():
    """Capture from one dispatcher, restore on another — simulates cross-process."""
    source_handler = PropagatingHandler()
    source = Dispatcher(span_handlers=[source_handler], propagate=False)

    token = active_instrument_tags.set({"env": "prod"})
    try:
        ctx = source.capture_propagation_context()
    finally:
        active_instrument_tags.reset(token)

    dest_handler = PropagatingHandler()
    dest = Dispatcher(span_handlers=[dest_handler], propagate=False)

    dest.restore_propagation_context(ctx)

    assert dest_handler._restored_context == ctx
    assert active_instrument_tags.get() == {"env": "prod"}
    # cleanup
    active_instrument_tags.set({})


def test_capture_swallows_handler_exceptions():
    class BrokenHandler(SimpleSpanHandler):
        def capture_propagation_context(self) -> Dict[str, Any]:
            raise RuntimeError("boom")

    handler = BrokenHandler()
    d = Dispatcher(span_handlers=[handler], propagate=False)

    # Should not raise
    ctx = d.capture_propagation_context()
    assert isinstance(ctx, dict)


def test_restore_swallows_handler_exceptions():
    class BrokenHandler(SimpleSpanHandler):
        def restore_propagation_context(self, context: Dict[str, Any]) -> None:
            raise RuntimeError("boom")

    handler = BrokenHandler()
    d = Dispatcher(span_handlers=[handler], propagate=False)

    # Should not raise
    d.restore_propagation_context({"some": "data"})
