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


# Tests for the deprecated asyncio_module() shim
# Co-authored-by: Hermes Agent <hermes-agent@nousresearch.com>

import asyncio as _asyncio
import warnings as _warnings

from llama_index.core.async_utils import asyncio_module


def test_asyncio_module_returns_asyncio_when_no_progress() -> None:
    """asyncio_module(show_progress=False) should return the standard asyncio module."""
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", DeprecationWarning)
        mod = asyncio_module(show_progress=False)
    assert mod is _asyncio


def test_asyncio_module_emits_deprecation_warning() -> None:
    """asyncio_module() must emit a DeprecationWarning pointing to get_asyncio_module."""
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        asyncio_module()
    assert len(caught) == 1
    w = caught[0]
    assert issubclass(w.category, DeprecationWarning)
    assert "get_asyncio_module" in str(w.message)
