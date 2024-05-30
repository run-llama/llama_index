import asyncio

import pytest

from llama_index.core.async_utils import asyncio_run


def test_asyncio_run() -> None:
    async def foo() -> int:
        return 0

    assert asyncio_run(foo()) == 0


def test_asyncio_run_existing_event_loop() -> None:
    async def foo(x: int) -> int:
        return asyncio_run(foo(x - 1)) if x else 0

    assert asyncio.run(foo(2)) == 0


def test_asyncio_run_with_exception() -> None:
    async def foo(x: int) -> int:
        if x:
            return asyncio_run(foo(x - 1))
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        assert asyncio_run(foo(0))

    with pytest.raises(asyncio.CancelledError):
        assert asyncio.run(foo(2))
