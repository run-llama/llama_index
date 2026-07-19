"""Tests for SQLJoinQueryEngine async behavior."""

import asyncio
import threading
import time
from unittest.mock import MagicMock

import pytest

from llama_index.core.base.response.schema import Response
from llama_index.core.query_engine.sql_join_query_engine import (
    SQLJoinQueryEngine,
)
from llama_index.core.schema import QueryBundle


def _make_engine() -> SQLJoinQueryEngine:
    """
    Create a SQLJoinQueryEngine bypassing __init__.

    The constructor requires real SQL query engines, which is heavyweight
    for a unit test. We only need to exercise ``_aquery``'s delegation to
    ``_query``, so we skip ``__init__`` and patch ``_query`` directly.
    """
    return object.__new__(SQLJoinQueryEngine)


@pytest.mark.asyncio
async def test_aquery_delegates_to_query() -> None:
    """``_aquery`` should return whatever ``_query`` returns."""
    engine = _make_engine()
    sentinel = Response(response="from sql")
    engine._query = MagicMock(return_value=sentinel)

    bundle = QueryBundle(query_str="SELECT 1")
    result = await engine._aquery(bundle)

    assert result is sentinel
    engine._query.assert_called_once_with(bundle)


@pytest.mark.asyncio
async def test_aquery_uses_to_thread() -> None:
    """``_aquery`` should run ``_query`` in a worker thread, not on the loop."""
    engine = _make_engine()
    loop_thread = threading.get_ident()

    captured: dict = {}

    def _fake_query(bundle: QueryBundle) -> Response:
        # Record the thread that actually executes _query.
        captured["thread"] = threading.get_ident()
        # Block to simulate synchronous I/O; if _aquery ran this on the
        # event loop thread the background task below would be delayed.
        time.sleep(0.05)
        return Response(response="ok")

    engine._query = MagicMock(side_effect=_fake_query)

    ran_at: dict = {}

    async def _background() -> None:
        await asyncio.sleep(0.01)
        ran_at["t"] = True

    bg_task = asyncio.create_task(_background())
    await asyncio.sleep(0)  # let the background task schedule

    result = await engine._aquery(QueryBundle(query_str="q"))
    await bg_task

    # The result is correct.
    assert isinstance(result, Response)
    # _query ran in a *different* thread than the event loop.
    assert captured["thread"] != loop_thread
    # The background task got to run while _query was blocking, proving
    # the event loop was not blocked.
    assert ran_at.get("t") is True


@pytest.mark.asyncio
async def test_aquery_does_not_block_event_loop() -> None:
    """A blocking ``_query`` must not stall the event loop."""
    engine = _make_engine()

    def _slow_query(bundle: QueryBundle) -> Response:
        time.sleep(0.05)
        return Response(response="ok")

    engine._query = MagicMock(side_effect=_slow_query)

    loop = asyncio.get_running_loop()
    start = loop.time()
    ran_at = None

    async def _background() -> None:
        nonlocal ran_at
        await asyncio.sleep(0.01)
        ran_at = loop.time()

    bg_task = asyncio.create_task(_background())
    await asyncio.sleep(0)
    await engine._aquery(QueryBundle(query_str="q"))
    await bg_task

    assert ran_at is not None, "background task never ran"
    assert (ran_at - start) < 0.04, (
        f"background task finished {ran_at - start:.3f}s after start "
        f"(expected < 0.04s) — the event loop was likely blocked"
    )
