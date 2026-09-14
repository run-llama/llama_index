import asyncio
import contextvars
import pytest
from llama_index.core.async_utils import (
    batch_gather,
    asyncio_run,
    run_async_tasks,
)


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


@pytest.mark.parametrize("show_progress", [False, True])
def test_run_async_tasks_success(show_progress: bool) -> None:
    """Test run_async_tasks returns outputs when all tasks succeed."""

    async def task(val: int) -> int:
        return val * 2

    coros = [task(1), task(2), task(3)]
    results = run_async_tasks(coros, show_progress=show_progress)
    assert results == [2, 4, 6]


@pytest.mark.parametrize("show_progress", [False, True])
def test_run_async_tasks_exception_propagation(show_progress: bool) -> None:
    """Test run_async_tasks faithfully propagates task exceptions regardless of show_progress."""

    async def task_ok() -> int:
        return 1

    async def task_fail() -> None:
        raise ValueError("ORIGINAL task failure")

    with pytest.raises(ValueError, match="ORIGINAL task failure"):
        run_async_tasks([task_ok(), task_fail()], show_progress=show_progress)
