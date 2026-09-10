import asyncio
import contextvars
import threading
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
async def test_asyncio_run_shuts_down_default_executor() -> None:
    """
    Verify that asyncio_run shuts down the event loop's default executor.

    When a coroutine calls loop.run_in_executor(None, ...), the loop lazily
    creates a default ThreadPoolExecutor.  asyncio_run must shut it down
    before closing the loop so that its worker threads are joined and do not
    leak.
    """
    executor_threads_before = {
        t for t in threading.enumerate() if "ThreadPoolExecutor" in t.name
    }

    async def use_default_executor() -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: "done")

    # Calling from inside a running loop triggers the loop.is_running() path,
    # which creates a new loop in a worker thread.
    result = asyncio_run(use_default_executor())
    assert result == "done"

    executor_threads_after = {
        t for t in threading.enumerate() if "ThreadPoolExecutor" in t.name
    }
    leaked = executor_threads_after - executor_threads_before
    assert not leaked, f"Default executor threads were not shut down: {leaked}"


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
