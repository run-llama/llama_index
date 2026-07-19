import asyncio
import contextvars
import logging

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


def test_run_async_tasks_tqdm_fallback_logs_warning(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the tqdm progress path fails, a warning should be logged."""

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("tqdm unavailable")

    import nest_asyncio

    monkeypatch.setattr(nest_asyncio, "apply", _boom)

    async def _echo(x: int) -> int:
        return x

    tasks = [_echo(1), _echo(2)]
    with caplog.at_level(logging.WARNING, logger="llama_index.core.async_utils"):
        results = run_async_tasks(tasks, show_progress=True)

    assert results == [1, 2]
    assert any(
        "tqdm-based progress bar is unavailable" in record.message
        and record.levelno == logging.WARNING
        for record in caplog.records
    ), [r.message for r in caplog.records]
