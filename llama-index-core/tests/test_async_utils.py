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


def test_run_async_tasks_propagates_task_errors_with_progress() -> None:
    """
    show_progress only controls the progress bar, so a failing task must raise
    its own exception instead of being masked by the progress-bar fallback.
    """

    async def fail() -> None:
        raise ValueError("original task failure")

    with pytest.raises(ValueError, match="original task failure"):
        run_async_tasks([fail()], show_progress=True)


def test_run_async_tasks_returns_results_with_progress() -> None:
    """The progress-bar path still returns the task results."""

    async def echo(n: int) -> int:
        return n

    assert run_async_tasks([echo(1), echo(2)], show_progress=True) == [1, 2]
