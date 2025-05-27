import asyncio
from typing import Any, Dict, List, Optional, Sequence

from llama_index.core.bridge.pydantic import Field
from llama_index.core.evaluation.base import BaseEvaluator, EvaluationResult
from llama_index.core.prompts.mixin import PromptDictType, PromptMixinType

from tonic_validate.metrics.answer_consistency_metric import (
    AnswerConsistencyMetric,
)
from tonic_validate.metrics.answer_similarity_metric import (
    AnswerSimilarityMetric,
)
from tonic_validate.metrics.augmentation_accuracy_metric import (
    AugmentationAccuracyMetric,
)
from tonic_validate.metrics.augmentation_precision_metric import (
    AugmentationPrecisionMetric,
)
from tonic_validate.metrics.retrieval_precision_metric import (
    RetrievalPrecisionMetric,
)
from tonic_validate.validate_scorer import ValidateScorer


class TonicValidateEvaluationResult(EvaluationResult):
    score_dict: Dict[str, float] = Field(None, description="Scores for each metric")


class TonicValidateEvaluator(BaseEvaluator):
    """
    Tonic Validate's validate scorer. Calculates all of Tonic Validate's metrics.

    See https://docs.tonic.ai/validate/ for more details.

    Args:
        metrics(List[Metric]): The metrics to use. Defaults to all of Tonic Validate's
            metrics.
        model_evaluator(str): The OpenAI service to use. Specifies the chat completion
            model to use as the LLM evaluator. Defaults to "gpt-4".

    """

    def __init__(
        self, metrics: Optional[List[Any]] = None, model_evaluator: str = "gpt-4"
    ):
        if metrics is None:
            metrics = [
                AnswerConsistencyMetric(),
                AnswerSimilarityMetric(),
                AugmentationAccuracyMetric(),
                AugmentationPrecisionMetric(),
                RetrievalPrecisionMetric(),
            ]

        self.metrics = metrics
        self.model_evaluator = model_evaluator
        self.validate_scorer = ValidateScorer(metrics, model_evaluator)

    def _calculate_average_score(self, run: Any) -> float:
        from tonic_validate.metrics.answer_similarity_metric import (
            AnswerSimilarityMetric,
        )

        ave_score = 0.0
        metric_cnt = 0
        for metric_name, score in run.overall_scores.items():
            if metric_name == AnswerSimilarityMetric.name:
                ave_score += score / 5
            else:
                ave_score += score
            metric_cnt += 1
        return ave_score / metric_cnt

    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        reference_response: Optional[str] = None,
        **kwargs: Any,
    ) -> TonicValidateEvaluationResult:
        from tonic_validate.classes.benchmark import BenchmarkItem
        from tonic_validate.classes.llm_response import LLMResponse

        benchmark_item = BenchmarkItem(question=query, answer=reference_response)

        llm_response = LLMResponse(
            llm_answer=response,
            llm_context_list=contexts,
            benchmark_item=benchmark_item,
        )

        responses = [llm_response]

        run = self.validate_scorer.score_run(responses)

        ave_score = self._calculate_average_score(run)

        return TonicValidateEvaluationResult(
            query=query,
            contexts=contexts,
            response=response,
            score=ave_score,
            score_dict=run.run_data[0].scores,
        )

    async def aevaluate_run(
        self,
        queries: List[str],
        responses: List[str],
        contexts_list: List[List[str]],
        reference_responses: List[str],
        **kwargs: Any,
    ) -> Any:
        """
        Evaluates a batch of responses.

        Returns a Tonic Validate Run object, which can be logged to the Tonic Validate
        UI. See https://docs.tonic.ai/validate/ for more details.
        """
        from tonic_validate.classes.benchmark import BenchmarkItem
        from tonic_validate.classes.llm_response import LLMResponse

        llm_responses = []

        for query, response, contexts, reference_response in zip(
            queries, responses, contexts_list, reference_responses
        ):
            benchmark_item = BenchmarkItem(question=query, answer=reference_response)

            llm_response = LLMResponse(
                llm_answer=response,
                llm_context_list=contexts,
                benchmark_item=benchmark_item,
            )

            llm_responses.append(llm_response)

        return self.validate_scorer.score_run(llm_responses)

    def evaluate_run(
        self,
        queries: List[str],
        responses: List[str],
        contexts_list: List[List[str]],
        reference_responses: List[str],
        **kwargs: Any,
    ) -> Any:
        """
        Evaluates a batch of responses.

        Returns a Tonic Validate Run object, which can be logged to the Tonic Validate
        UI. See https://docs.tonic.ai/validate/ for more details.
        """
        return asyncio.run(
            self.aevaluate_run(
                queries=queries,
                responses=responses,
                contexts_list=contexts_list,
                reference_responses=reference_responses,
                **kwargs,
            )
        )

    def _get_prompts(self) -> PromptDictType:
        return {}

    def _get_prompt_modules(self) -> PromptMixinType:
        return {}

    def _update_prompts(self, prompts_dict: PromptDictType) -> None:
        return
