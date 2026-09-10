import asyncio
import contextvars
import pytest
from llama_index.core.async_utils import batch_gather, asyncio_run, run_async_tasks


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


def test_run_async_tasks_propagates_progress_task_errors() -> None:
    """
    A task error must propagate when show_progress=True.

    Previously the progress path wrapped both the optional tqdm/nest_asyncio
    import and task execution in a bare ``except Exception: pass``, so an
    exception raised by a task was swallowed and the call returned silently.
    """

    async def _raise_task_error() -> None:
        raise ValueError("task failed")

    with pytest.raises(ValueError, match="task failed"):
        run_async_tasks([_raise_task_error()], show_progress=True)


def test_run_async_tasks_falls_back_when_progress_loop_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A failure to set up the tqdm/nest_asyncio loop must not fail the call.

    Only the optional imports are allowed to trigger the fallback, but the loop
    setup itself can also fail (for example when nest_asyncio cannot patch an
    already-running loop). That case must degrade to a plain gather instead of
    propagating, while a task error still surfaces - see the test above.
    """
    nest_asyncio = pytest.importorskip("nest_asyncio")
    pytest.importorskip("tqdm.asyncio")

    def _boom() -> None:
        raise RuntimeError("cannot patch a running loop")

    monkeypatch.setattr(nest_asyncio, "apply", _boom)

    async def _ok() -> str:
        return "ok"

    assert run_async_tasks([_ok()], show_progress=True) == ["ok"]
