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


# ---------------------------------------------------------------------------
# Tests for get_asyncio_module
# ---------------------------------------------------------------------------

def test_get_asyncio_module_default_returns_asyncio() -> None:
    """get_asyncio_module() with no arguments should return the asyncio module."""
    from llama_index.core.async_utils import get_asyncio_module

    module = get_asyncio_module()
    assert module is asyncio


def test_get_asyncio_module_show_progress_false_returns_asyncio() -> None:
    """get_asyncio_module(show_progress=False) should return the asyncio module."""
    from llama_index.core.async_utils import get_asyncio_module

    module = get_asyncio_module(show_progress=False)
    assert module is asyncio


def test_get_asyncio_module_show_progress_true_returns_tqdm() -> None:
    """get_asyncio_module(show_progress=True) should return tqdm_asyncio."""
    pytest.importorskip("tqdm")
    from tqdm.asyncio import tqdm_asyncio
    from llama_index.core.async_utils import get_asyncio_module

    module = get_asyncio_module(show_progress=True)
    assert module is tqdm_asyncio


def test_get_asyncio_module_has_gather() -> None:
    """The returned module must expose a 'gather' attribute in both modes."""
    from llama_index.core.async_utils import get_asyncio_module

    assert hasattr(get_asyncio_module(show_progress=False), "gather")
