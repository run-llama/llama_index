import asyncio
from llama_index.core.async_utils import batch_gather


def test_batch_gather_uneven_task_list() -> None:
    """
    Test that batch_gather works with an uneven task list.
    """

    async def async_method():
        return 1

    coroutines = [async_method() for _ in range(5)]
    results = asyncio.run(batch_gather(coroutines, batch_size=2))
    assert results == [1] * len(coroutines)
