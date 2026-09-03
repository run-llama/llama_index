"""Tests for MultiStepQueryEngine async correctness."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_index.core.base.response.schema import Response
from llama_index.core.callbacks import CallbackManager
from llama_index.core.query_engine.multistep_query_engine import MultiStepQueryEngine
from llama_index.core.schema import QueryBundle


def _make_engine(num_steps: int = 2) -> MultiStepQueryEngine:
    inner_engine = MagicMock()
    inner_engine.callback_manager = CallbackManager([])
    resp = Response(response="sub-answer", source_nodes=[])
    inner_engine.query.return_value = resp
    inner_engine.aquery = AsyncMock(return_value=resp)

    transform = MagicMock()
    transform.return_value = QueryBundle(query_str="sub-question")

    synth = MagicMock()
    synth.synthesize.return_value = Response(response="final")
    synth.asynthesize = AsyncMock(return_value=Response(response="final"))

    return MultiStepQueryEngine(
        query_engine=inner_engine,
        query_transform=transform,
        response_synthesizer=synth,
        num_steps=num_steps,
        early_stopping=True,
    )


def test_query_uses_sync_engine():
    engine = _make_engine(num_steps=1)
    result = engine.query(QueryBundle(query_str="What is X?"))
    assert str(result) == "final"
    engine._query_engine.query.assert_called_once()
    engine._query_engine.aquery.assert_not_awaited()


@pytest.mark.asyncio
async def test_aquery_uses_async_engine():
    """_aquery must call aquery on the inner engine, never the sync query."""
    engine = _make_engine(num_steps=1)
    result = await engine.aquery(QueryBundle(query_str="What is X?"))
    assert str(result) == "final"
    engine._query_engine.aquery.assert_awaited_once()
    engine._query_engine.query.assert_not_called()


@pytest.mark.asyncio
async def test_aquery_multi_steps_uses_async_engine():
    """With multiple steps, every step must go through aquery."""
    engine = _make_engine(num_steps=3)
    result = await engine.aquery(QueryBundle(query_str="What is X?"))
    assert str(result) == "final"
    assert engine._query_engine.aquery.await_count == 3
    engine._query_engine.query.assert_not_called()
