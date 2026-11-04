import threading
from unittest.mock import MagicMock

import pytest

from llama_index.core.base.response.schema import Response
from llama_index.core.query_engine.retry_query_engine import (
    RetryGuidelineQueryEngine,
    RetryQueryEngine,
)
from llama_index.core.schema import QueryBundle


@pytest.mark.asyncio
@pytest.mark.parametrize("engine_class", [RetryQueryEngine, RetryGuidelineQueryEngine])
async def test_retry_aquery_runs_sync_query_off_event_loop(engine_class, monkeypatch):
    event_loop_thread = threading.get_ident()
    query_thread = None
    expected = Response(response="ok")
    bundle = QueryBundle(query_str="test")

    def fake_query(self, query_bundle):
        nonlocal query_thread
        query_thread = threading.get_ident()
        assert query_bundle is bundle
        return expected

    monkeypatch.setattr(engine_class, "_query", fake_query)
    if engine_class is RetryQueryEngine:
        engine = engine_class(query_engine=MagicMock(), evaluator=MagicMock())
    else:
        engine = engine_class(
            query_engine=MagicMock(),
            guideline_evaluator=MagicMock(),
            query_transformer=MagicMock(),
        )

    assert await engine._aquery(bundle) is expected
    assert query_thread is not None
    assert query_thread != event_loop_thread
