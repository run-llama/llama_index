import inspect

from llama_index_instrumentation.span_handlers.simple import SimpleSpanHandler


def _bound_args() -> inspect.BoundArguments:
    return inspect.signature(lambda: None).bind()


def test_span_exit_unknown_id_is_a_no_op() -> None:
    """Exiting a span that is not open must not raise, mirroring span_drop()."""
    handler = SimpleSpanHandler()

    # The drop path already tolerates an unknown id.
    handler.span_drop(id_="never-opened", bound_args=_bound_args())

    # The exit path must behave the same way instead of raising KeyError.
    handler.span_exit(id_="never-opened", bound_args=_bound_args())

    assert handler.open_spans == {}
    assert handler.completed_spans == []


def test_span_exit_after_span_drop_is_a_no_op() -> None:
    """A span dropped during shutdown must not blow up if a late exit arrives."""
    handler = SimpleSpanHandler()
    handler.span_enter(id_="s1", bound_args=_bound_args())
    assert "s1" in handler.open_spans

    handler.span_drop(id_="s1", bound_args=_bound_args())
    handler.span_exit(id_="s1", bound_args=_bound_args())

    assert handler.open_spans == {}
    assert len(handler.completed_spans) == 0
    assert len(handler.dropped_spans) == 1


def test_span_exit_still_completes_open_spans() -> None:
    """The normal path is unchanged: an open span is completed and removed."""
    handler = SimpleSpanHandler()
    handler.span_enter(id_="s1", bound_args=_bound_args())

    handler.span_exit(id_="s1", bound_args=_bound_args())

    assert handler.open_spans == {}
    assert len(handler.completed_spans) == 1
