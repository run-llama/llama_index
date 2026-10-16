"""Tests that sync-fallback ``_aquery`` implementations offload to a thread."""

import asyncio
import threading
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from llama_index.core.base.response.schema import Response
from llama_index.core.query_engine.flare.base import FLAREInstructQueryEngine
from llama_index.core.query_engine.graph_query_engine import (
    ComposableGraphQueryEngine,
)
from llama_index.core.query_engine.retry_query_engine import (
    RetryGuidelineQueryEngine,
    RetryQueryEngine,
)
from llama_index.core.query_engine.retry_source_query_engine import (
    RetrySourceQueryEngine,
)
from llama_index.core.query_engine.router_query_engine import (
    RetrieverRouterQueryEngine,
)
from llama_index.core.schema import QueryBundle

# (engine class, sync method that _aquery offloads to a worker thread)
ENGINES: list[tuple[type, str]] = [
    (RetryQueryEngine, "_query"),
    (RetryGuidelineQueryEngine, "_query"),
    (RetrySourceQueryEngine, "_query"),
    (FLAREInstructQueryEngine, "_query"),
    (RetrieverRouterQueryEngine, "_query"),
    (ComposableGraphQueryEngine, "_query_index"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("engine_cls", "patch_target"), ENGINES)
async def test_aquery_delegates_to_sync_method(
    engine_cls: type, patch_target: str
) -> None:
    """``_aquery`` should return whatever the sync method returns."""
    engine: Any = object.__new__(engine_cls)
    sentinel = Response(response="ok")
    sync_mock: MagicMock = MagicMock(return_value=sentinel)
    setattr(engine, patch_target, sync_mock)

    bundle = QueryBundle(query_str="q")
    result = await engine._aquery(bundle)

    assert result is sentinel
    sync_mock.assert_called_once()
    assert sync_mock.call_args[0][0] is bundle


@pytest.mark.asyncio
@pytest.mark.parametrize(("engine_cls", "patch_target"), ENGINES)
async def test_aquery_does_not_block_event_loop(
    engine_cls: type, patch_target: str
) -> None:
    """A blocking sync method must run in a worker thread, not on the loop."""
    engine: Any = object.__new__(engine_cls)
    loop_thread = threading.get_ident()
    captured: dict[str, int] = {}

    def _slow_sync_method(bundle: QueryBundle, *args: Any, **kwargs: Any) -> Response:
        # Record the thread that actually executes the sync method.
        captured["thread"] = threading.get_ident()
        # Simulate synchronous I/O; if _aquery ran this on the event loop
        # thread the background task below could not run concurrently.
        time.sleep(0.05)
        return Response(response="ok")

    setattr(engine, patch_target, _slow_sync_method)

    ran_at: dict[str, bool] = {}

    async def _background() -> None:
        await asyncio.sleep(0.01)
        ran_at["t"] = True

    bg_task = asyncio.create_task(_background())
    await asyncio.sleep(0)  # let the background task schedule

    result = await engine._aquery(QueryBundle(query_str="q"))
    await bg_task

    assert isinstance(result, Response)
    # The sync method ran in a *different* thread than the event loop.
    assert captured["thread"] != loop_thread
    # The background task got to run while the sync method was blocking,
    # proving the event loop was not blocked.
    assert ran_at.get("t") is True
