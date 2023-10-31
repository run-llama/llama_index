"""Correctness evaluation."""
from typing import Any, Optional, Sequence, Union

from llama_index.evaluation.base import BaseEvaluator, EvaluationResult
from llama_index.indices.service_context import ServiceContext
from llama_index.prompts import (
    BasePromptTemplate,
    ChatMessage,
    ChatPromptTemplate,
    MessageRole,
    PromptTemplate,
)
from llama_index.prompts.mixin import PromptDictType

DEFAULT_SYSTEM_TEMPLATE = """
You are an expert evaluation system for a question answering chatbot.

You are given the following information:
- a user query,
- a reference answer, and
- a generated answer.

Your job is to judge the relevance and correctness of the generated answer.
Output a single score that represents a holistic evaluation.
You must return your response in a line with only the score.
Do not return answers in any other format.
On a separate line provide your reasoning for the score as well.

Follow these guidelines for scoring:
- Your score has to be between 1 and 5, where 1 is the worst and 5 is the best.
- If the generated answer is not relevant to the user query, \
you should give a score of 1.
- If the generated answer is relevant but contains mistakes, \
you should give a score between 2 and 3.
- If the generated answer is relevant and fully correct, \
you should give a score between 4 and 5.

Example Response:
4.0
The generated answer has the exact same metrics as the reference answer, \
    but it is not as concise.

"""

DEFAULT_USER_TEMPLATE = """
## User Query
{query}

## Reference Answer
{reference_answer}

## Generated Answer
{generated_answer}
"""

DEFAULT_EVAL_TEMPLATE = ChatPromptTemplate(
    message_templates=[
        ChatMessage(role=MessageRole.SYSTEM, content=DEFAULT_SYSTEM_TEMPLATE),
        ChatMessage(role=MessageRole.USER, content=DEFAULT_USER_TEMPLATE),
    ]
)


class CorrectnessEvaluator(BaseEvaluator):
    """Correctness evaluator.

    Evaluates the correctness of a question answering system.
    This evaluator depends on `reference` answer to be provided, in addition to the
    query string and response string.

    It outputs a score between 1 and 5, where 1 is the worst and 5 is the best,
    along with a reasoning for the score.
    Passing is defined as a score greater than or equal to the given threshold.

    Args:
        service_context (Optional[ServiceContext]): Service context.
        eval_template (Optional[Union[BasePromptTemplate, str]]):
            Template for the evaluation prompt.
        score_threshold (float): Numerical threshold for passing the evaluation,
            defaults to 4.0.
    """

    def __init__(
        self,
        service_context: Optional[ServiceContext] = None,
        eval_template: Optional[Union[BasePromptTemplate, str]] = None,
        score_threshold: float = 4.0,
    ) -> None:
        self._service_context = service_context or ServiceContext.from_defaults()

        self._eval_template: BasePromptTemplate
        if isinstance(eval_template, str):
            self._eval_template = PromptTemplate(eval_template)
        else:
            self._eval_template = eval_template or DEFAULT_EVAL_TEMPLATE

        self._score_threshold = score_threshold

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {
            "eval_template": self._eval_template,
        }

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "eval_template" in prompts:
            self._eval_template = prompts["eval_template"]

    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        del kwargs  # Unused
        del contexts  # Unused

        if query is None or response is None or reference is None:
            print(query, response, reference, flush=True)
            raise ValueError("query, response, and reference must be provided")

        eval_response = await self._service_context.llm_predictor.apredict(
            prompt=self._eval_template,
            query=query,
            generated_answer=response,
            reference_answer=reference,
        )

        # Extract from response
        score_str, reasoning_str = eval_response.split("\n", 1)
        score = float(score_str)
        reasoning = reasoning_str.lstrip("\n")

        return EvaluationResult(
            query=query,
            response=response,
            passing=score >= self._score_threshold,
            score=score,
            feedback=reasoning,
        )
