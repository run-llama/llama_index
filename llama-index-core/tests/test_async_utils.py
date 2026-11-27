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


def test_asyncio_run_keeps_the_error_raised_by_the_coroutine() -> None:
    """The coroutine's own RuntimeError must reach the caller, not be reported as a nested async problem."""

    async def failing() -> None:
        raise RuntimeError("the real cause")

    with pytest.raises(RuntimeError, match="the real cause"):
        asyncio_run(failing())

    captured = {}

    def run_in_thread() -> None:
        try:
            asyncio_run(failing())
        except RuntimeError as e:
            captured["message"] = str(e)

    thread = threading.Thread(target=run_in_thread)
    thread.start()
    thread.join()
    assert captured["message"] == "the real cause"
