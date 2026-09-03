from typing import Any, Optional, Sequence

import pytest

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import Response
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.evaluation.base import BaseEvaluator, EvaluationResult
from llama_index.core.prompts.mixin import PromptDictType, PromptMixinType
from llama_index.core.query_engine.retry_query_engine import RetryGuidelineQueryEngine
from llama_index.core.schema import QueryBundle


class FakeQueryEngine(BaseQueryEngine):
    def __init__(self, response: Response) -> None:
        super().__init__(callback_manager=CallbackManager([]))
        self._response = response

    def _get_prompt_modules(self) -> PromptMixinType:
        return {}

    def _query(self, query_bundle: QueryBundle) -> Response:
        return self._response

    async def _aquery(self, query_bundle: QueryBundle) -> Response:
        return self._response


class RecordingEvaluator(BaseEvaluator):
    def __init__(self) -> None:
        self.responses: list[Optional[str]] = []

    def _get_prompts(self) -> PromptDictType:
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        pass

    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        self.responses.append(response)
        return EvaluationResult(
            query=query,
            response=response,
            contexts=contexts,
            passing=True,
            feedback="ok",
        )


def test_retry_guideline_evaluates_sql_query_metadata_instead_of_raw_rows() -> None:
    raw_rows = "RAW_ROW\n" * 1000
    query_engine = FakeQueryEngine(
        Response(response=raw_rows, metadata={"sql_query": "SELECT id FROM users"})
    )
    evaluator = RecordingEvaluator()
    engine = RetryGuidelineQueryEngine(query_engine, evaluator)

    response = engine.query("show user IDs")

    assert response.response == raw_rows
    assert evaluator.responses == ["SELECT id FROM users"]


def test_retry_guideline_preserves_normal_response_evaluation() -> None:
    query_engine = FakeQueryEngine(Response(response="normal answer"))
    evaluator = RecordingEvaluator()
    engine = RetryGuidelineQueryEngine(query_engine, evaluator)

    engine.query("normal question")

    assert evaluator.responses == ["normal answer"]


@pytest.mark.asyncio
async def test_retry_guideline_async_uses_sql_query_metadata_for_evaluation() -> None:
    raw_rows = "RAW_ROW\n" * 1000
    query_engine = FakeQueryEngine(
        Response(response=raw_rows, metadata={"sql_query": "SELECT id FROM users"})
    )
    evaluator = RecordingEvaluator()
    engine = RetryGuidelineQueryEngine(query_engine, evaluator)

    response = await engine.aquery("show user IDs")

    assert response.response == raw_rows
    assert evaluator.responses == ["SELECT id FROM users"]
