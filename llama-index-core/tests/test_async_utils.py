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


from llama_index.core.async_utils import chunks


def test_chunks_even_split() -> None:
    """chunks() splits an evenly-divisible iterable into equal-size tuples."""
    result = list(chunks([1, 2, 3, 4], 2))
    assert result == [(1, 2), (3, 4)]


def test_chunks_with_padding() -> None:
    """chunks() pads the last tuple with None when the iterable length is not
    a multiple of the chunk size."""
    result = list(chunks([1, 2, 3, 4, 5], 2))
    assert result == [(1, 2), (3, 4), (5, None)]


def test_chunks_size_larger_than_iterable() -> None:
    """chunks() returns a single padded tuple when size > len(iterable)."""
    result = list(chunks([1, 2], 5))
    assert result == [(1, 2, None, None, None)]


def test_chunks_empty_iterable() -> None:
    """chunks() yields nothing for an empty iterable."""
    result = list(chunks([], 3))
    assert result == []
