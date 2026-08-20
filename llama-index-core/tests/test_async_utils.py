import asyncio
import contextvars
from typing import Coroutine

import pytest
from llama_index.core.async_utils import batch_gather, asyncio_run


def test_batch_gather_indivisible_task_list() -> None:
    """
    Test that batch_gather works with an task list of a
    length that is not cleanly divisible by the batch size.
    """

    async def async_method(n: int) -> int:
        return n

    coroutines = [async_method(n) for n in range(5)]
    results = asyncio.run(batch_gather(coroutines, batch_size=2))
    assert results == list(range(len(coroutines)))


@pytest.mark.asyncio
async def test_asyncio_run_copies_contextvars_when_loop_running() -> None:
    """
    Validate that context vars are copied when loop.is_running() is True.
    """
    test_var: contextvars.ContextVar[str] = contextvars.ContextVar(
        "test_var", default=""
    )
    token = test_var.set("sentinel_value")
    try:

        async def read_context() -> str:
            return test_var.get()

        # Calling from inside a running loop triggers the loop.is_running() path
        result = asyncio_run(read_context())
        assert result == "sentinel_value"
    finally:
        test_var.reset(token)


def test_asyncio_run_propagates_coroutine_runtime_error() -> None:
    """A RuntimeError raised by the coroutine itself must propagate unchanged."""

    async def fail() -> None:
        raise RuntimeError("original failure")

    with pytest.raises(RuntimeError, match="original failure"):
        asyncio_run(fail())


def test_asyncio_run_propagates_runtime_error_in_no_loop_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The no-event-loop fallback must not mask the coroutine's RuntimeError."""

    async def fail() -> None:
        raise RuntimeError("fallback failure")

    def raise_no_loop() -> None:
        raise RuntimeError("no current event loop in thread")

    monkeypatch.setattr(asyncio, "get_event_loop", raise_no_loop)
    with pytest.raises(RuntimeError, match="fallback failure"):
        asyncio_run(fail())


def test_asyncio_run_no_loop_fallback_runs_coroutine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no current event loop, the coroutine still runs via asyncio.run()."""

    async def answer() -> int:
        return 42

    def raise_no_loop() -> None:
        raise RuntimeError("no current event loop in thread")

    monkeypatch.setattr(asyncio, "get_event_loop", raise_no_loop)
    assert asyncio_run(answer()) == 42


def test_asyncio_run_nested_async_still_gets_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuine nested-async failure keeps the actionable error message."""

    async def noop() -> None:
        return None

    def raise_no_loop() -> None:
        raise RuntimeError("no current event loop in thread")

    def raise_nested(coro: Coroutine) -> None:
        coro.close()
        raise RuntimeError("asyncio.run() cannot be called from a running event loop")

    monkeypatch.setattr(asyncio, "get_event_loop", raise_no_loop)
    monkeypatch.setattr(asyncio, "run", raise_nested)
    # Simulate a loop genuinely running in this thread (the nested-async case).
    monkeypatch.setattr(
        "llama_index.core.async_utils._is_loop_running_in_this_thread",
        lambda: True,
    )
    with pytest.raises(RuntimeError, match="Detected nested async"):
        asyncio_run(noop())
