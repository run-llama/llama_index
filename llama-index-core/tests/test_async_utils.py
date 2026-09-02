import asyncio
import contextvars
import pytest
from llama_index.core.async_utils import (
    batch_gather,
    asyncio_run,
    gather_with_bounded_lifetime,
)


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


@pytest.mark.asyncio
async def test_gather_with_bounded_lifetime_all_succeed() -> None:
    """All coroutines succeed: results are returned in call order."""

    async def async_method(n: int) -> int:
        return n

    coros = [async_method(n) for n in range(5)]
    results = await gather_with_bounded_lifetime(coros)
    assert results == list(range(5))


@pytest.mark.asyncio
async def test_gather_with_bounded_lifetime_empty_list() -> None:
    """An empty coroutine list returns an empty list without calling gather_fn."""
    results = await gather_with_bounded_lifetime([])
    assert results == []


@pytest.mark.asyncio
async def test_gather_with_bounded_lifetime_waits_for_siblings_before_raising() -> None:
    """
    Regression test for #22312: a single failing coroutine must not leave
    sibling coroutines orphaned in the background. By the time this function
    raises, every coroutine it started must have already finished.
    """
    completed: list[str] = []

    async def worker(name: str, should_fail: bool, delay: float) -> str:
        if should_fail:
            await asyncio.sleep(0.01)
            raise ValueError(f"failed: {name}")
        await asyncio.sleep(delay)
        completed.append(name)
        return name

    coros = [
        worker("a", False, 0.2),
        worker("b", True, 0),
        worker("c", False, 0.2),
    ]

    with pytest.raises(ValueError, match="failed: b"):
        await gather_with_bounded_lifetime(coros)

    # If this were a bare asyncio.gather, the ValueError would propagate
    # almost immediately (after ~0.01s), well before "a" and "c" (0.2s
    # delay) finish - leaving them orphaned. Asserting they're already in
    # `completed` proves gather_with_bounded_lifetime waited for them.
    assert sorted(completed) == ["a", "c"]


@pytest.mark.asyncio
async def test_gather_with_bounded_lifetime_raises_first_exception() -> None:
    """When multiple coroutines fail, the first one (by position) is raised."""

    async def fail(msg: str, delay: float) -> None:
        await asyncio.sleep(delay)
        raise ValueError(msg)

    coros = [fail("first", 0), fail("second", 0.05)]

    with pytest.raises(ValueError, match="first"):
        await gather_with_bounded_lifetime(coros)


@pytest.mark.asyncio
async def test_gather_with_bounded_lifetime_custom_gather_fn() -> None:
    """A custom gather_fn (e.g. tqdm_asyncio.gather) is used and receives kwargs."""
    received_kwargs = {}

    async def custom_gather(*coros, **kwargs):
        received_kwargs.update(kwargs)
        return await asyncio.gather(*coros)

    async def async_method(n: int) -> int:
        return n

    coros = [async_method(n) for n in range(3)]
    results = await gather_with_bounded_lifetime(
        coros, gather_fn=custom_gather, desc="test", total=3
    )
    assert results == [0, 1, 2]
    assert received_kwargs == {"desc": "test", "total": 3}


@pytest.mark.asyncio
async def test_gather_with_bounded_lifetime_reraises_cancelled_error() -> None:
    """CancelledError must propagate immediately, not be captured as a result."""

    async def cancels_self() -> None:
        raise asyncio.CancelledError

    async def normal() -> int:
        await asyncio.sleep(0.05)
        return 1

    with pytest.raises(asyncio.CancelledError):
        await gather_with_bounded_lifetime([cancels_self(), normal()])
