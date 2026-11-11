from typing import Any, Optional, Sequence

from llama_index.core.base.response.schema import NodeWithScore, Response
from llama_index.core.evaluation import BaseEvaluator
from llama_index.core.evaluation.base import EvaluationResult
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.schema import TextNode


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
        **kwargs: Any,
    ) -> EvaluationResult:
        return EvaluationResult(
            query=query,
            contexts=contexts,
            response=response,
            passing=self._mock_passing,
            score=self._mock_score,
            feedback=self._mock_feedback,
        )


def test_evaluator_basic() -> None:
    test_evaluator = MockEvaluator()
    eval_result_0 = test_evaluator.evaluate(
        query="test query",
        response="test response",
        contexts=["test context 1", "test context 2"],
    )

    eval_result_1 = test_evaluator.evaluate_response(
        query="test query",
        response=Response(
            response="test response",
            source_nodes=[
                NodeWithScore(node=TextNode(text="test context 1"), score=1.0),
                NodeWithScore(node=TextNode(text="test context 2"), score=1.0),
            ],
        ),
    )

    assert eval_result_0 == eval_result_1


def test_guideline_evaluator_prompt_isolation_and_update() -> None:
    from llama_index.core.evaluation.guideline import (
        DEFAULT_EVAL_TEMPLATE,
        EvaluationData,
        GuidelineEvaluator,
    )
    from llama_index.core.llms.mock import MockLLM
    from llama_index.core.prompts import PromptTemplate

    assert DEFAULT_EVAL_TEMPLATE.output_parser is None

    custom_template = PromptTemplate(
        "Query: {query}\nResponse: {response}\nGuidelines: {guidelines}\n"
    )
    evaluator = GuidelineEvaluator(llm=MockLLM(), eval_template=custom_template)

    # Initializing GuidelineEvaluator must not mutate DEFAULT_EVAL_TEMPLATE or caller template
    assert DEFAULT_EVAL_TEMPLATE.output_parser is None
    assert custom_template.output_parser is None
    assert evaluator.get_prompts()["eval_template"].output_parser is not None

    default_evaluator = GuidelineEvaluator(llm=MockLLM())
    assert DEFAULT_EVAL_TEMPLATE.output_parser is None
    assert default_evaluator.get_prompts()["eval_template"].output_parser is not None

    # Updating eval_template without an output_parser should preserve the evaluator's output_parser
    replacement_template = PromptTemplate(
        "Updated Query: {query}\nResponse: {response}\nGuidelines: {guidelines}\n"
    )
    default_evaluator.update_prompts({"eval_template": replacement_template})
    assert replacement_template.output_parser is None
    updated_parser = default_evaluator.get_prompts()["eval_template"].output_parser
    assert updated_parser is not None
    assert updated_parser.output_cls is EvaluationData

