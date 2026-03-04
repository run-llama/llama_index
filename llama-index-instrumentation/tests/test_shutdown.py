import inspect
from unittest.mock import MagicMock

from llama_index_instrumentation.dispatcher import Dispatcher, Manager
from llama_index_instrumentation.span_handlers.simple import SimpleSpanHandler


def _make_bound_args():
    return inspect.signature(lambda: None).bind()


def test_shutdown_drops_all_open_spans():
    handler = SimpleSpanHandler()
    d = Dispatcher(span_handlers=[handler], propagate=False)

    # Open some spans
    for i in range(3):
        handler.span_enter(
            id_=f"span-{i}",
            bound_args=_make_bound_args(),
            parent_id=None,
        )

    assert len(handler.open_spans) == 3

    d.shutdown()

    assert len(handler.open_spans) == 0
    assert len(handler.dropped_spans) == 3


def test_shutdown_calls_close_on_handlers():
    close_mock = MagicMock()

    class TrackingHandler(SimpleSpanHandler):
        def close(self) -> None:
            close_mock()

    handler = TrackingHandler()
    d = Dispatcher(span_handlers=[handler], propagate=False)

    d.shutdown()

    close_mock.assert_called_once()


def test_shutdown_walks_parent_chain():
    parent_handler = SimpleSpanHandler()
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

    # Open spans on both handlers
    parent_handler.span_enter(id_="p-span", bound_args=_make_bound_args())
    child_handler.span_enter(id_="c-span", bound_args=_make_bound_args())

    child.shutdown()

    assert len(parent_handler.open_spans) == 0
    assert len(child_handler.open_spans) == 0
    assert len(parent_handler.dropped_spans) == 1
    assert len(child_handler.dropped_spans) == 1


def test_shutdown_is_idempotent():
    handler = SimpleSpanHandler()
    d = Dispatcher(span_handlers=[handler], propagate=False)

    handler.span_enter(id_="span-1", bound_args=_make_bound_args())
    d.shutdown()
    d.shutdown()  # should not error

    assert len(handler.open_spans) == 0
    assert len(handler.dropped_spans) == 1


def test_close_default_is_noop():
    handler = SimpleSpanHandler()
    handler.close()  # should not raise
