from typing import Any, Optional, Sequence

from llama_index.core.evaluation.base import BaseEvaluator, EvaluationResult
from llama_index.core.prompts.mixin import PromptDictType, PromptMixinType

from tonic_validate.metrics.answer_similarity_metric import (
    AnswerSimilarityMetric,
)
from tonic_validate.services.openai_service import OpenAIService


class AnswerSimilarityEvaluator(BaseEvaluator):
    """
    Tonic Validate's answer similarity metric.

    The output score is a float between 0.0 and 5.0.

    See https://docs.tonic.ai/validate/ for more details.

    Args:
        openai_service(OpenAIService): The OpenAI service to use. Specifies the chat
            completion model to use as the LLM evaluator. Defaults to "gpt-4".

    """

    def __init__(self, openai_service: Optional[Any] = None):
        if openai_service is None:
            openai_service = OpenAIService("gpt-4")
        self.openai_service = openai_service
        self.metric = AnswerSimilarityMetric()

    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        reference_response: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        from tonic_validate.classes.benchmark import BenchmarkItem
        from tonic_validate.classes.llm_response import LLMResponse

        benchmark_item = BenchmarkItem(question=query, answer=reference_response)

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

    def _get_prompt_modules(self) -> PromptMixinType:
        return {}

    def _update_prompts(self, prompts_dict: PromptDictType) -> None:
        return
