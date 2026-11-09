from unittest.mock import MagicMock

import pytest

from llama_index.core import Settings
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import Response
from llama_index.core.evaluation import EvaluationResult, GuidelineEvaluator
from llama_index.core.indices.query.query_transform.feedback_transform import (
    FeedbackQueryTransformation,
)
from llama_index.core.llms.mock import MockLLM
from llama_index.core.query_engine import RetryGuidelineQueryEngine
from llama_index.core.schema import QueryBundle


@pytest.mark.parametrize("max_retries", [0, 1, 2, 3])
def test_retry_guideline_preserves_custom_query_transformer(monkeypatch, max_retries):
    monkeypatch.setattr(Settings, "_llm", MockLLM())
    query_engine = MagicMock(spec=BaseQueryEngine)
    query_engine._query.return_value = Response(response="Initial answer")
    evaluator = MagicMock(spec=GuidelineEvaluator)
    evaluator.evaluate_response.return_value = EvaluationResult(
        passing=False, response="Initial answer", feedback="Try again"
    )
    transformer = MagicMock(spec=FeedbackQueryTransformation)
    transformer.run.side_effect = lambda query, metadata: QueryBundle(
        query.query_str + " transformed"
    )
    engine = RetryGuidelineQueryEngine(
        query_engine,
        evaluator,
        query_transformer=transformer,
        max_retries=max_retries,
    )

    engine.query("Question")

    assert transformer.run.call_count == max_retries
    assert [call.args[0].query_str for call in query_engine._query.call_args_list] == [
        "Question" + " transformed" * index for index in range(max_retries + 1)
    ]
