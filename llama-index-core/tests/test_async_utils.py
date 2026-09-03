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


def test_run_async_tasks_propagates_progress_task_errors() -> None:
    """
    A task error must propagate when show_progress=True.

    Previously the progress path wrapped both the optional tqdm/nest_asyncio
    import and task execution in a bare ``except Exception: pass``, so an
    exception raised by a task was swallowed and the call returned silently.
    """

    async def _raise_task_error() -> None:
        raise ValueError("task failed")

    with pytest.raises(ValueError, match="task failed"):
        run_async_tasks([_raise_task_error()], show_progress=True)
