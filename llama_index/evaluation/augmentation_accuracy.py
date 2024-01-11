from typing import Any, Optional, Sequence

from llama_index.evaluation.base import BaseEvaluator, EvaluationResult
from tonic_validate.metrics.augmentation_accuracy_metric import (
    AugmentationAccuracyMetric,
)
from tonic_validate.services.openai_service import OpenAIService
from tonic_validate.classes.llm_response import LLMResponse
from tonic_validate.classes.benchmark_item import BenchmarkItem


class AugmentationAccuracyEvaluator(BaseEvaluator):
    def __init__(self, openai_service: OpenAIService = OpenAIService("gpt-4")):
        self.openai_service = openai_service
        self.metric = AugmentationAccuracyMetric()

    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        **kwargs: Any
    ) -> EvaluationResult:
        benchmark_item = BenchmarkItem(question=query, answer=response)

        llm_response = LLMResponse(
            llm_answer=response,
            llm_context_list=contexts,
            benchmark_item=benchmark_item,
        )

        score = self.metric.score(llm_response, self.openai_service)

        return EvaluationResult(
            query=query, contexts=contexts, response=response, score=score
        )
