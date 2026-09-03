import asyncio
import contextvars
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


def test_asyncio_run_preserves_runtime_error_from_coroutine() -> None:
    async def fail() -> None:
        raise RuntimeError("original failure")

    with pytest.raises(RuntimeError, match="original failure"):
        asyncio_run(fail())


def test_asyncio_run_preserves_runtime_error_in_no_loop_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail() -> None:
        raise RuntimeError("fallback failure")

    def raise_no_loop() -> None:
        raise RuntimeError("no current event loop")

    monkeypatch.setattr(asyncio, "get_event_loop", raise_no_loop)

    with pytest.raises(RuntimeError, match="fallback failure"):
        asyncio_run(fail())


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


@pytest.mark.asyncio
async def test_asyncio_run_preserves_runtime_error_when_loop_running() -> None:
    async def fail() -> None:
        raise RuntimeError("threaded failure")

    with pytest.raises(RuntimeError, match="threaded failure"):
        asyncio_run(fail())
