import warnings

import pytest

from llama_index_instrumentation.span.simple import SimpleSpan
from llama_index_instrumentation.span_handlers.simple import SimpleSpanHandler


def _handler_with(spans) -> SimpleSpanHandler:
    handler = SimpleSpanHandler()
    handler.completed_spans = list(spans)
    return handler


def _call_trace_trees(handler: SimpleSpanHandler) -> None:
    """Trace-tree rendering warns about missing parents; that is not under test here."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        handler.print_trace_trees()


def test_trace_trees_does_not_rewrite_span_parent_ids() -> None:
    """Rendering the trees is a read-only view and must not mutate stored spans."""
    pytest.importorskip("treelib")
    span = SimpleSpan(id_="child", parent_id="ghost-parent")
    handler = _handler_with([span])

    _call_trace_trees(handler)
    assert span.parent_id == "ghost-parent"


def test_trace_trees_is_idempotent_across_calls() -> None:
    """Repeated rendering must not keep appending to the parent id."""
    pytest.importorskip("treelib")
    span = SimpleSpan(id_="child", parent_id="ghost-parent")
    handler = _handler_with([span])

    for _ in range(3):
        _call_trace_trees(handler)

    assert span.parent_id == "ghost-parent"
    assert "MISSING" not in span.parent_id
    assert len(handler.completed_spans) == 1
