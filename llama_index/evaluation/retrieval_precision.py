from typing import Any, Optional, Sequence

from llama_index.evaluation.base import BaseEvaluator, EvaluationResult
from llama_index.prompts.mixin import PromptDictType
from tonic_validate.metrics.retrieval_precision_metric import (
    RetrievalPrecisionMetric,
)
from tonic_validate.services.openai_service import OpenAIService
from tonic_validate.classes.llm_response import LLMResponse
from tonic_validate.classes.benchmark import BenchmarkItem


class RetrievalPrecisionEvaluator(BaseEvaluator):
    def __init__(self, openai_service: OpenAIService = OpenAIService("gpt-4")):
        self.openai_service = openai_service
        self.metric = RetrievalPrecisionMetric()

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

    def _get_prompts(self) -> PromptDictType:
        return {}

    def _get_prompt_modules(self) -> PromptDictType:
        return {}

    def _update_prompts(self, prompts_dict: PromptDictType) -> None:
        return
