import asyncio

import pytest

from llama_index.core.async_utils import batch_gather


@pytest.mark.parametrize("batch_size", [0, -1])
@pytest.mark.parametrize("with_tasks", [False, True])
def test_batch_gather_rejects_nonpositive_size(
    batch_size: int, with_tasks: bool
) -> None:
    calls = []

    async def work() -> int:
        calls.append("ran")
        return 1

    tasks = [work()] if with_tasks else []
    try:
        with pytest.raises(ValueError, match="batch_size must be greater than zero"):
            asyncio.run(batch_gather(tasks, batch_size=batch_size))
        assert calls == []
    finally:
        for task in tasks:
            task.close()


@pytest.mark.parametrize("batch_size", [1, 2, 10])
def test_batch_gather_preserves_results_and_order(batch_size: int) -> None:
    async def work(value: int) -> int:
        return value

    assert asyncio.run(
        batch_gather([work(i) for i in range(5)], batch_size=batch_size)
    ) == list(range(5))
    assert asyncio.run(batch_gather([], batch_size=batch_size)) == []
