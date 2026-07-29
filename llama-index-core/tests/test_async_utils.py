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


@pytest.mark.parametrize("show_progress", [False, True])
def test_run_async_tasks_propagates_task_exception(show_progress: bool) -> None:
    """
    `show_progress` is cosmetic and must not change error semantics.

    The tqdm branch used to catch every exception -- including ones raised by
    the tasks themselves -- and then fall through to re-await the already
    consumed coroutines, so a real task failure surfaced as an unrelated
    "Detected nested async" RuntimeError.
    """

    async def ok() -> int:
        return 1

    async def fail() -> None:
        raise ValueError("original task failure")

    with pytest.raises(ValueError, match="original task failure"):
        run_async_tasks([ok(), fail()], show_progress=show_progress)


@pytest.mark.parametrize("show_progress", [False, True])
def test_run_async_tasks_returns_results(show_progress: bool) -> None:
    """Results must not depend on whether a progress bar is shown."""

    async def one() -> int:
        return 1

    async def two() -> int:
        return 2

    assert run_async_tasks([one(), two()], show_progress=show_progress) == [1, 2]


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
