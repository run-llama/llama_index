from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_index.core.base.response.schema import Response
from llama_index.core.callbacks import CallbackManager
from llama_index.core.query_engine.multistep_query_engine import MultiStepQueryEngine
from llama_index.core.schema import QueryBundle


def _make_query_engine():
    engine = MagicMock()
    engine.callback_manager = CallbackManager([])
    resp = Response(response="answer", source_nodes=[])
    engine.query.return_value = resp
    engine.aquery = AsyncMock(return_value=resp)
    return engine


def _make_query_transform():
    transform = MagicMock()
    transform.side_effect = lambda query_bundle, metadata=None: QueryBundle(
        query_str="step query"
    )
    return transform


def test_query_calls_sync_query_engine():
    engine = _make_query_engine()
    mse = MultiStepQueryEngine(
        query_engine=engine,
        query_transform=_make_query_transform(),
        num_steps=1,
    )

    mse.query("hello")

    engine.query.assert_called_once()
    engine.aquery.assert_not_called()


@pytest.mark.asyncio
async def test_aquery_calls_async_query_engine():
    """
    Regression test: aquery() must await the wrapped engine's aquery(),
    not silently fall back to its synchronous query().
    """
    engine = _make_query_engine()
    mse = MultiStepQueryEngine(
        query_engine=engine,
        query_transform=_make_query_transform(),
        num_steps=1,
    )

    await mse.aquery("hello")

    engine.aquery.assert_awaited_once()
    engine.query.assert_not_called()
