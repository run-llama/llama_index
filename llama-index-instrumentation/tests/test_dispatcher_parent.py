import inspect

from llama_index_instrumentation.dispatcher import Dispatcher, Manager
from llama_index_instrumentation.span_handlers.simple import SimpleSpanHandler


def _bound_args() -> inspect.BoundArguments:
    return inspect.signature(lambda: None).bind()


def test_parent_is_none_without_a_manager() -> None:
    """A standalone dispatcher has no parent instead of raising AssertionError."""
    assert Dispatcher(name="solo").parent is None


def test_parent_is_none_when_parent_name_is_not_registered() -> None:
    """A root dispatcher's parent_name is empty and resolves to None, not KeyError."""
    root = Dispatcher(name="root")
    manager = Manager(root=root)
    root.manager = manager

    assert root.parent is None


def test_log_name_works_on_a_standalone_dispatcher() -> None:
    """log_name() must not require a manager."""
    assert Dispatcher(name="solo").log_name == "solo"


def test_log_name_uses_parent_when_resolvable() -> None:
    root = Dispatcher(name="root")
    manager = Manager(root=root)
    root.manager = manager
    child = Dispatcher(name="child", parent_name="root", manager=manager)
    manager.add_dispatcher(child)

    assert child.log_name == "root.child"
    assert root.log_name == "root"


def test_shutdown_works_on_a_standalone_dispatcher() -> None:
    """shutdown() is a public API and must work without a manager."""
    dispatcher = Dispatcher(name="solo")
    dispatcher.add_span_handler(SimpleSpanHandler())

    dispatcher.shutdown()


def test_shutdown_walks_the_chain_without_a_registered_grandparent() -> None:
    """The propagation walk ends at the root instead of raising KeyError."""
    root = Dispatcher(name="root")
    manager = Manager(root=root)
    root.manager = manager
    child = Dispatcher(name="child", parent_name="root", manager=manager)
    manager.add_dispatcher(child)

    handler = SimpleSpanHandler()
    child.add_span_handler(handler)
    child.span_enter(id_="s1", bound_args=_bound_args())

    child.shutdown()

    assert handler.open_spans == {}
    assert len(handler.dropped_spans) == 1
