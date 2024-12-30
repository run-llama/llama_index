from typing import Any, Optional, Sequence

from llama_index.core.base.response.schema import Response
from llama_index.core.evaluation import BaseEvaluator
from llama_index.core.evaluation.base import EvaluationResult
from llama_index.core.prompts.mixin import PromptDictType

from llama_index.core.evaluation.batch_runner import BatchEvalRunner


class MockEvaluator(BaseEvaluator):
    def __init__(
        self,
        mock_score: float = 1.0,
        mock_passing: bool = True,
        mock_feedback: str = "test feedback",
    ) -> None:
        self._mock_score = mock_score
        self._mock_passing = mock_passing
        self._mock_feedback = mock_feedback

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""

    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        return EvaluationResult(
            query=query,
            contexts=contexts,
            response=response,
            passing=(
                str(response) == str(reference) if reference else self._mock_passing
            ),
            score=self._mock_score,
            feedback=self._mock_feedback,
        )


def get_eval_results(key, eval_results):
    results = eval_results[key]
    correct = 0
    for result in results:
        if result.passing:
            correct += 1
    return correct / len(results)


def test_batch_runner() -> None:
    # single evaluator
    runner = BatchEvalRunner(
        evaluators={
            "evaluator1": MockEvaluator(),
            "no_kwarg_evaluator": MockEvaluator(),
        }
    )

    exp_queries = ["query1", "query2"]
    exp_response_strs = ["response1", "response2"]
    exp_responses = [
        Response(response="response1", source_nodes=[]),
        Response(response="response2", source_nodes=[]),
    ]
    # original eval_kwargs_lists format - Dict[str, List]
    exp_kwargs = {"reference": ["response1", "response1"]}

    # test evaluate_response_strs()
    results = runner.evaluate_response_strs(
        queries=exp_queries, response_strs=exp_response_strs, **exp_kwargs
    )
    assert get_eval_results("evaluator1", results) == 0.5

    # test evaluate_responses()
    results = runner.evaluate_responses(
        queries=exp_queries, responses=exp_responses, **exp_kwargs
    )
    assert get_eval_results("evaluator1", results) == 0.5

    # multiple evaluators
    runner.evaluators = {
        "evaluator1": MockEvaluator(),
        "evaluator2": MockEvaluator(),
        "no_kwarg_evaluator": MockEvaluator(),
    }

    exp_queries = ["query1", "query2"]
    exp_response_strs = ["response1", "response2"]
    exp_responses = [
        Response(response="response1", source_nodes=[]),
        Response(response="response2", source_nodes=[]),
    ]
    # updated eval_kwargs_lists format - Dict[str, Dict[str, List]]
    exp_kwargs = {
        "evaluator1": {"reference": ["response1", "response1"]},
        "evaluator2": {"reference": ["response1", "response2"]},
    }

    # test evaluate_response_strs()
    results = runner.evaluate_response_strs(
        queries=exp_queries, response_strs=exp_response_strs, **exp_kwargs
    )
    assert get_eval_results("evaluator1", results) == 0.5
    assert get_eval_results("evaluator2", results) == 1.0

    # test evaluate_responses()
    results = runner.evaluate_responses(
        queries=exp_queries, responses=exp_responses, **exp_kwargs
    )
    assert get_eval_results("evaluator1", results) == 0.5
    assert get_eval_results("evaluator2", results) == 1.0
    assert get_eval_results("evaluator1", results) == 0.5
    assert get_eval_results("evaluator2", results) == 1.0
