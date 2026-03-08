"""Regression tests for SubQuestionQueryEngine partial-failure handling.

Issue #20904: _query_subq / _aquery_subq previously only caught ValueError.
Common runtime exceptions (RuntimeError, KeyError, TimeoutError, etc.) escaped
uncaught, making the entire query fail instead of skipping the broken sub-
question and continuing with the remaining results.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llama_index.core.base.response.schema import Response
from llama_index.core.callbacks import CallbackManager
from llama_index.core.query_engine.sub_question_query_engine import (
    SubQuestionQueryEngine,
)
from llama_index.core.question_gen.types import SubQuestion
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.schema import QueryBundle
from llama_index.core.tools import QueryEngineTool, ToolMetadata


# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------


class _AlwaysOkEngine:
    """Query engine that always returns a fixed response."""

    def __init__(self, answer: str = "ok") -> None:
        self._answer = answer
        self.callback_manager = CallbackManager([])

    def query(self, query: Any) -> Response:
        return Response(response=self._answer)

    async def aquery(self, query: Any) -> Response:
        return Response(response=self._answer)


class _AlwaysFailEngine:
    """Query engine that always raises the given exception type."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc
        self.callback_manager = CallbackManager([])

    def query(self, query: Any) -> Response:
        raise self._exc

    async def aquery(self, query: Any) -> Response:
        raise self._exc


def _make_engine(tools: list[QueryEngineTool]) -> SubQuestionQueryEngine:
    question_gen = MagicMock()
    return SubQuestionQueryEngine(
        question_gen=question_gen,
        response_synthesizer=get_response_synthesizer(),
        query_engine_tools=tools,
        use_async=False,
        verbose=False,
    )


def _make_tool(name: str, engine: Any) -> QueryEngineTool:
    return QueryEngineTool(
        query_engine=engine,
        metadata=ToolMetadata(name=name, description=name),
    )


# ---------------------------------------------------------------------------
# Sync tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("API rate limit exceeded"),
        KeyError("missing_key"),
        TimeoutError("timed out"),
        ConnectionError("network failure"),
    ],
    ids=["RuntimeError", "KeyError", "TimeoutError", "ConnectionError"],
)
def test_query_subq_tolerates_non_value_error(exc: BaseException) -> None:
    """_query_subq must return None (not re-raise) for any exception type."""
    engine = _make_engine(
        [_make_tool("ok_tool", _AlwaysOkEngine()), _make_tool("fail_tool", _AlwaysFailEngine(exc))]
    )

    # Call _query_subq directly — should return None, not raise.
    result = engine._query_subq(
        SubQuestion(sub_question="irrelevant?", tool_name="fail_tool")
    )
    assert result is None


def test_batch_query_skips_failed_sub_question() -> None:
    """The overall query must succeed even if one sub-engine raises RuntimeError."""
    ok_engine = _AlwaysOkEngine(answer="Paris")
    fail_engine = _AlwaysFailEngine(RuntimeError("rate limit"))

    tools = [
        _make_tool("france_docs", ok_engine),
        _make_tool("germany_docs", fail_engine),
    ]
    engine = _make_engine(tools)

    sub_questions = [
        SubQuestion(sub_question="Capital of France?", tool_name="france_docs"),
        SubQuestion(sub_question="Capital of Germany?", tool_name="germany_docs"),
    ]

    with patch.object(engine._question_gen, "generate", return_value=sub_questions):
        # Should not raise; partial results should be synthesised.
        response = engine.query("Capitals of France and Germany?")

    assert response is not None


# ---------------------------------------------------------------------------
# Async tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("API rate limit exceeded"),
        KeyError("missing_key"),
        TimeoutError("timed out"),
    ],
    ids=["RuntimeError", "KeyError", "TimeoutError"],
)
async def test_aquery_subq_tolerates_non_value_error(exc: BaseException) -> None:
    """_aquery_subq must return None (not re-raise) for any exception type."""
    engine = _make_engine(
        [_make_tool("ok_tool", _AlwaysOkEngine()), _make_tool("fail_tool", _AlwaysFailEngine(exc))]
    )

    result = await engine._aquery_subq(
        SubQuestion(sub_question="irrelevant?", tool_name="fail_tool")
    )
    assert result is None
