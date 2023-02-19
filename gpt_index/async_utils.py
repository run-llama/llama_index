"""Async utils."""
import asyncio
from typing import Any, Coroutine, List


def run_async_tasks(tasks: List[Coroutine]) -> List[Any]:
    """Run async tasks."""
    all_tasks = asyncio.gather(*tasks)
    outputs: List[Any] = asyncio.run(all_tasks)  # type: ignore
    return outputs
