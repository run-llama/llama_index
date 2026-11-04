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


def test_asyncio_module_deprecation_warning() -> None:
    """asyncio_module() should emit a DeprecationWarning."""
    import warnings
    from llama_index.core.async_utils import asyncio_module

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio_module()

    assert any(issubclass(w.category, DeprecationWarning) for w in caught), (
        "Expected a DeprecationWarning from asyncio_module()"
    )


def test_asyncio_module_returns_asyncio_by_default() -> None:
    """asyncio_module(show_progress=False) should return the asyncio module."""
    import warnings
    from llama_index.core.async_utils import asyncio_module

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        mod = asyncio_module(show_progress=False)

    assert mod is asyncio
