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


def test_run_async_tasks_propagates_exceptions_with_show_progress() -> None:
    """
    Regression test for #22493: task exceptions must propagate regardless of
    show_progress.  The broad `except Exception: pass` that swallowed them has
    been replaced with a narrow `except ImportError` that only catches a missing
    tqdm/nest_asyncio installation.
    """

    async def ok() -> int:
        return 1

    async def fail() -> int:
        raise ValueError("ORIGINAL task failure")

    # Without progress bar — baseline, must raise the original error.
    with pytest.raises(ValueError, match="ORIGINAL task failure"):
        run_async_tasks([ok(), fail()], show_progress=False)

    # With progress bar — must raise the same error, not a misleading one.
    with pytest.raises(ValueError, match="ORIGINAL task failure"):
        run_async_tasks([ok(), fail()], show_progress=True)


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
