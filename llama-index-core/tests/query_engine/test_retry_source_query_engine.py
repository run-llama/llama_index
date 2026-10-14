from unittest.mock import MagicMock

import pytest

from llama_index.core import Settings
from llama_index.core.base.response.schema import Response
from llama_index.core.evaluation import BaseEvaluator, EvaluationResult
from llama_index.core.llms.mock import MockLLM
from llama_index.core.query_engine import RetrieverQueryEngine, RetrySourceQueryEngine
from llama_index.core.schema import NodeWithScore, TextNode


@pytest.mark.parametrize("passing", [False, None])
def test_retry_source_excludes_nodes_that_do_not_pass(monkeypatch, passing):
    llm = MockLLM()
    monkeypatch.setattr(Settings, "_llm", llm)
    query_engine = MagicMock(spec=RetrieverQueryEngine)
    query_engine._query.return_value = Response(
        response="Initial answer",
        source_nodes=[
            NodeWithScore(node=TextNode(text="Accepted source")),
            NodeWithScore(node=TextNode(text="Rejected source")),
        ],
    )
    evaluator = MagicMock(spec=BaseEvaluator)
    evaluator.evaluate_response.return_value = EvaluationResult(passing=False)
    evaluator.evaluate.side_effect = [
        EvaluationResult(passing=True),
        EvaluationResult(passing=passing),
    ]
    engine = RetrySourceQueryEngine(query_engine, evaluator, llm=llm, max_retries=1)

    response = engine.query("Which source should be used?")

    assert [node.get_content() for node in response.source_nodes] == ["Accepted source"]


@pytest.mark.parametrize("passing", [False, None])
def test_retry_source_raises_when_no_nodes_pass(passing):
    query_engine = MagicMock(spec=RetrieverQueryEngine)
    query_engine._query.return_value = Response(
        response="Initial answer",
        source_nodes=[NodeWithScore(node=TextNode(text="Rejected source"))],
    )
    evaluator = MagicMock(spec=BaseEvaluator)
    evaluator.evaluate_response.return_value = EvaluationResult(passing=False)
    evaluator.evaluate.return_value = EvaluationResult(passing=passing)
    engine = RetrySourceQueryEngine(
        query_engine, evaluator, llm=MockLLM(), max_retries=1
    )

    with pytest.raises(ValueError, match="No source nodes passed evaluation"):
        engine.query("Which source should be used?")
