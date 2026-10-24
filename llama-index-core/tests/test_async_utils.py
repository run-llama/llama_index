import asyncio
import contextvars
import gc
import inspect
from typing import Any, Coroutine
import pytest
from llama_index.core.async_utils import batch_gather, asyncio_run, run_jobs


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


async def _finish_test_jobs(
    release: asyncio.Event,
    previous_tasks: set[asyncio.Task[Any]],
    jobs: list[Coroutine[Any, Any, Any]],
) -> None:
    release.set()
    await asyncio.gather(
        *(asyncio.all_tasks() - previous_tasks), return_exceptions=True
    )
    for job in jobs:
        job.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("show_progress", [False, True])
@pytest.mark.parametrize("error_type", [ValueError, asyncio.CancelledError])
async def test_run_jobs_cleans_up_after_job_failure(
    show_progress: bool, error_type: type[BaseException]
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    cleaned_up = asyncio.Event()
    error = error_type("job failed")

    async def fail() -> None:
        await started.wait()
        raise error

    async def wait() -> None:
        started.set()
        try:
            await release.wait()
        finally:
            await asyncio.sleep(0)
            cleaned_up.set()

    jobs = [fail(), wait()]
    previous_tasks = asyncio.all_tasks()
    try:
        with pytest.raises(error_type) as exc_info:
            await run_jobs(jobs, workers=2, show_progress=show_progress)
        assert exc_info.value is error
        assert cleaned_up.is_set()
        assert all(
            inspect.getcoroutinestate(job) == inspect.CORO_CLOSED for job in jobs
        )
        assert not (asyncio.all_tasks() - previous_tasks)
    finally:
        await _finish_test_jobs(release, previous_tasks, jobs)


@pytest.mark.asyncio
@pytest.mark.parametrize("show_progress", [False, True])
async def test_run_jobs_cleans_up_after_caller_cancellation(
    show_progress: bool,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    cleaned_up = asyncio.Event()

    async def wait() -> None:
        started.set()
        try:
            await release.wait()
        finally:
            await asyncio.sleep(0)
            cleaned_up.set()

    jobs = [wait(), wait(), wait()]
    previous_tasks = asyncio.all_tasks()
    task = asyncio.create_task(run_jobs(jobs, workers=1, show_progress=show_progress))
    try:
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cleaned_up.is_set()
        assert all(
            inspect.getcoroutinestate(job) == inspect.CORO_CLOSED for job in jobs
        )
        assert not (asyncio.all_tasks() - previous_tasks)
    finally:
        await _finish_test_jobs(release, previous_tasks, jobs)


@pytest.mark.asyncio
@pytest.mark.parametrize("show_progress", [False, True])
@pytest.mark.parametrize("job_fails", [False, True])
async def test_run_jobs_finishes_cleanup_when_caller_cancels_again(
    show_progress: bool, job_fails: bool
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    cleaned_up = asyncio.Event()

    async def fail() -> None:
        await started.wait()
        raise ValueError("job failed")

    async def wait() -> None:
        started.set()
        try:
            await release.wait()
        finally:
            cleanup_started.set()
            await cleanup_release.wait()
            cleaned_up.set()

    jobs = [fail(), wait()] if job_fails else [wait(), wait(), wait()]
    previous_tasks = asyncio.all_tasks()
    task = asyncio.create_task(
        run_jobs(jobs, workers=2 if job_fails else 1, show_progress=show_progress)
    )
    try:
        if not job_fails:
            await started.wait()
            task.cancel()
        cleanup_waiter = asyncio.create_task(cleanup_started.wait())
        await asyncio.wait({task, cleanup_waiter}, return_when=asyncio.FIRST_COMPLETED)
        assert cleanup_started.is_set()
        task.cancel()
        await asyncio.sleep(0)
        cleanup_release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cleaned_up.is_set()
        assert all(
            inspect.getcoroutinestate(job) == inspect.CORO_CLOSED for job in jobs
        )
        assert not (asyncio.all_tasks() - previous_tasks)
    finally:
        cleanup_release.set()
        await _finish_test_jobs(release, previous_tasks, jobs)
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("show_progress", [False, True])
async def test_run_jobs_retrieves_exceptions_after_cancellation(
    show_progress: bool,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def wait() -> None:
        started.set()
        await release.wait()

    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    errors: list[dict[str, Any]] = []
    loop.set_exception_handler(lambda _, context: errors.append(context))
    jobs = [wait(), wait()]
    previous_tasks = asyncio.all_tasks()
    task = asyncio.create_task(run_jobs(jobs, show_progress=show_progress))
    try:
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)
        gc.collect()
        assert not errors
    finally:
        await _finish_test_jobs(release, previous_tasks, jobs)
        loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
@pytest.mark.parametrize("show_progress", [False, True])
async def test_run_jobs_preserves_order_and_worker_limit(show_progress: bool) -> None:
    started = [asyncio.Event() for _ in range(3)]
    started_indices: list[int] = []
    two_started = asyncio.Event()
    release = [asyncio.Event() for _ in range(3)]
    completed = [asyncio.Event() for _ in range(3)]
    running = 0
    max_running = 0

    async def job(index: int) -> int:
        nonlocal running, max_running
        running += 1
        max_running = max(max_running, running)
        started_indices.append(index)
        started[index].set()
        if running == 2:
            two_started.set()
        try:
            await release[index].wait()
            return index
        finally:
            running -= 1
            completed[index].set()

    task = asyncio.create_task(
        run_jobs([job(i) for i in range(3)], workers=2, show_progress=show_progress)
    )
    try:
        await two_started.wait()
        first, second = started_indices
        third = next(index for index in range(3) if index not in started_indices)
        assert not started[third].is_set()
        release[second].set()
        await completed[second].wait()
        await started[third].wait()
        release[third].set()
        await completed[third].wait()
        release[first].set()
        assert await task == [0, 1, 2]
        assert max_running == 2
        assert running == 0
        assert await run_jobs([], show_progress=show_progress) == []
    finally:
        for event in release:
            event.set()
        await task
