import asyncio
from typing import Any, Coroutine, List

DEFAULT_NUM_WORKERS = 4


def get_asyncio_module(show_progress: bool = False) -> Any:
    if show_progress:
        from tqdm.asyncio import tqdm_asyncio

        module = tqdm_asyncio
    else:
        module = asyncio

    return module


async def run_jobs(
    jobs: List[Coroutine],
    show_progress: bool = False,
    workers: int = DEFAULT_NUM_WORKERS,
) -> List[Any]:
    """Run jobs.

    Args:
        jobs (List[Coroutine]):
            List of jobs to run.
        show_progress (bool):
            Whether to show progress bar.

    Returns:
        List[Any]:
            List of results.
    """
    asyncio_mod = get_asyncio_module(show_progress=show_progress)
    semaphore = asyncio.Semaphore(workers)

    async def worker(job: Coroutine) -> Any:
        async with semaphore:
            return await job

    pool_jobs = [worker(job) for job in jobs]

    return await asyncio_mod.gather(*pool_jobs)
