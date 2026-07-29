import asyncio
import contextvars
import pytest
from llama_index.core.async_utils import batch_gather, asyncio_run, run_async_tasks


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


def test_run_async_tasks_propagates_exceptions_with_progress() -> None:
    """
    Regression test: run_async_tasks(show_progress=True) must not swallow
    task exceptions. Previously the broad ``except Exception: pass`` around
    the tqdm progress path also caught task errors, then the fallback
    re-awaited already-consumed coroutines and raised a misleading
    'Detected nested async' RuntimeError.

    See: https://github.com/run-llama/llama_index/issues/22493
    """

    async def ok() -> int:
        return 1

    async def fail() -> int:
        raise ValueError("ORIGINAL task failure")

    # show_progress=False already works correctly
    with pytest.raises(ValueError, match="ORIGINAL task failure"):
        run_async_tasks([ok(), fail()], show_progress=False)

    # show_progress=True must raise the same original exception
    with pytest.raises(ValueError, match="ORIGINAL task failure"):
        run_async_tasks([ok(), fail()], show_progress=True)


def test_run_async_tasks_success_with_progress() -> None:
    """run_async_tasks with show_progress=True returns correct results."""

    async def add_one(n: int) -> int:
        return n + 1

    coroutines = [add_one(n) for n in range(5)]
    results = run_async_tasks(coroutines, show_progress=True)
    assert results == [1, 2, 3, 4, 5]
