"""Async utils."""

import asyncio
import contextvars
import concurrent.futures
from itertools import zip_longest
from typing import Any, Callable, Coroutine, Iterable, List, Optional, TypeVar

import llama_index.core.instrumentation as instrument

dispatcher = instrument.get_dispatcher(__name__)


def get_asyncio_module(show_progress: bool = False) -> Any:
    if show_progress:
        from tqdm.asyncio import tqdm_asyncio

        module = tqdm_asyncio
    else:
        module = asyncio

    return module


def asyncio_module(show_progress: bool = False) -> Any:
    import warnings

    warnings.warn(
        "asyncio_module() is deprecated and will be removed in a future release. "
        "Use get_asyncio_module() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_asyncio_module(show_progress=show_progress)


def asyncio_run(coro: Coroutine) -> Any:
    """
    Gets an existing event loop to run the coroutine.

    If there is no existing event loop, creates a new one.
    If an event loop is already running, uses threading to run in a separate thread.
    """
    try:
        # Check if there's an existing event loop
        loop = asyncio.get_event_loop()

        # Check if the loop is already running
        if loop.is_running():
            # If loop is already running, run in a separate thread
            # Snapshot the current context so we can propagate contextvars
            ctx = contextvars.copy_context()

            def run_coro_in_thread() -> Any:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return ctx.run(new_loop.run_until_complete, coro)
                finally:
                    new_loop.close()

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_coro_in_thread)
                return future.result()
        else:
            # If we're here, there's an existing loop but it's not running
            return loop.run_until_complete(coro)

    except RuntimeError as e:
        # If we can't get the event loop, we're likely in a different thread
        try:
            return asyncio.run(coro)
        except RuntimeError as e:
            raise RuntimeError(
                "Detected nested async. Please use nest_asyncio.apply() to allow nested event loops."
                "Or, use async entry methods like `aquery()`, `aretriever`, `achat`, etc."
            )


def run_async_tasks(
    tasks: List[Coroutine],
    show_progress: bool = False,
    progress_bar_desc: str = "Running async tasks",
) -> List[Any]:
    """Run a list of async tasks."""
    tasks_to_execute: List[Any] = tasks
    if show_progress:
        try:
            import nest_asyncio
            from tqdm.asyncio import tqdm

            # jupyter notebooks already have an event loop running
            # we need to reuse it instead of creating a new one
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()

            async def _tqdm_gather() -> List[Any]:
                return await tqdm.gather(*tasks_to_execute, desc=progress_bar_desc)

            tqdm_outputs: List[Any] = loop.run_until_complete(_tqdm_gather())
            return tqdm_outputs
        # run the operation w/o tqdm on hitting a fatal
        # may occur in some environments where tqdm.asyncio
        # is not supported
        except Exception:
            pass

    async def _gather() -> List[Any]:
        return await asyncio.gather(*tasks_to_execute)

    outputs: List[Any] = asyncio_run(_gather())
    return outputs


def chunks(iterable: Iterable, size: int) -> Iterable:
    args = [iter(iterable)] * size
    return zip_longest(*args, fillvalue=None)


async def batch_gather(
    tasks: List[Coroutine], batch_size: int = 10, verbose: bool = False
) -> List[Any]:
    output: List[Any] = []
    for task_chunk in chunks(tasks, batch_size):
        task_chunk = (task for task in task_chunk if task is not None)
        output_chunk = await asyncio.gather(*task_chunk)
        output.extend(output_chunk)
        if verbose:
            print(f"Completed {len(output)} out of {len(tasks)} tasks")
    return output


DEFAULT_NUM_WORKERS = 4

T = TypeVar("T")


async def _await_or_return_exception(coro: Coroutine[Any, Any, T]) -> Any:
    """
    Await a coroutine, returning any raised exception instead of propagating it.

    Used by ``gather_with_bounded_lifetime`` to ensure that when several
    coroutines are run concurrently, a single failure does not leave sibling
    coroutines running unsupervised in the background. ``asyncio.CancelledError``
    is re-raised immediately rather than captured, to preserve cooperative
    cancellation semantics.
    """
    try:
        return await coro
    except asyncio.CancelledError:
        raise
    except BaseException as e:
        return e


async def gather_with_bounded_lifetime(
    coros: List[Coroutine[Any, Any, T]],
    gather_fn: Callable[..., Coroutine[Any, Any, List[Any]]] = asyncio.gather,
    **gather_kwargs: Any,
) -> List[T]:
    """
    Run coroutines concurrently via ``gather_fn`` without orphaning siblings on failure.

    A bare ``asyncio.gather(*coros)`` raises as soon as one coroutine fails,
    while any other coroutines that were already scheduled keep running
    detached in the background with nothing left to await or cancel them. This
    function instead waits for every coroutine to actually finish before
    raising, so the lifetime of the whole batch stays bounded to the call that
    spawned it: by the time this function returns (or raises), nothing it
    started is still running.

    Args:
        coros: List of coroutines to run concurrently.
        gather_fn: The gather implementation to use, e.g. ``asyncio.gather``
            (the default) or ``tqdm.asyncio.tqdm_asyncio.gather``. Must accept
            the wrapped coroutines as positional arguments plus arbitrary
            keyword arguments, and return results in call order.
        **gather_kwargs: Additional keyword arguments forwarded to
            ``gather_fn`` (e.g. ``desc``/``total`` for a tqdm-based gather).

    Returns:
        List of results, in the same order as ``coros``.

    Raises:
        The first exception raised by any coroutine in ``coros``, once all
        coroutines have completed.

    """
    if not coros:
        return []

    wrapped = [_await_or_return_exception(c) for c in coros]
    results = await gather_fn(*wrapped, **gather_kwargs)

    for result in results:
        if isinstance(result, BaseException):
            raise result

    return results


@dispatcher.span
async def run_jobs(
    jobs: List[Coroutine[Any, Any, T]],
    show_progress: bool = False,
    workers: int = DEFAULT_NUM_WORKERS,
    desc: Optional[str] = None,
) -> List[T]:
    """
    Run jobs.

    Args:
        jobs (List[Coroutine]):
            List of jobs to run.
        show_progress (bool):
            Whether to show progress bar.

    Returns:
        List[Any]:
            List of results.

    """
    semaphore = asyncio.Semaphore(workers)

    @dispatcher.span
    async def worker(job: Coroutine) -> Any:
        async with semaphore:
            return await job

    pool_jobs = [worker(job) for job in jobs]

    if show_progress:
        from tqdm.asyncio import tqdm_asyncio

        results = await tqdm_asyncio.gather(*pool_jobs, desc=desc)
    else:
        results = await asyncio.gather(*pool_jobs)

    return results
