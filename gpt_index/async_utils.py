"""Async utils."""
import asyncio
from typing import Any, Coroutine, List


def run_async_tasks(tasks: List[Coroutine]) -> List[Any]:
    """Run a list of async tasks."""

    async def _gather() -> List[Any]:
        return await asyncio.gather(*tasks)

    outputs: List[Any] = asyncio.run(_gather())
    return outputs
