"""Pairwise evaluation."""

from typing import Any, Optional, Sequence, Union

from llama_index import ServiceContext
from llama_index.evaluation.base import BaseEvaluator, EvaluationResult
from llama_index.prompts import (
    BasePromptTemplate,
    ChatMessage,
    ChatPromptTemplate,
    MessageRole,
    PromptTemplate,
)

DEFAULT_SYSTEM_TEMPLATE = """
You are an expert evaluation system for a question answering chatbot.

You are given the following information:
- a user query,
- Answer 1
- Answer 2


Your job is to output whether Answer 1 is better, or Answer 2 is better, or
they are equally good at answering the user query.

Output "1" if Answer 1 is better, "2" if Answer 2 is better, and \
    "TIE" if they are equally good.

Please output two lines:
- first line: "1", "2", or "TIE"
- second line: a short explanation for your decision.
"""

DEFAULT_USER_TEMPLATE = """
## User Query
{query}

## Answer 1
{reference_answer}

## Answer 2
{generated_answer}
"""

DEFAULT_EVAL_TEMPLATE = ChatPromptTemplate(
    message_templates=[
        ChatMessage(role=MessageRole.SYSTEM, content=DEFAULT_SYSTEM_TEMPLATE),
        ChatMessage(role=MessageRole.USER, content=DEFAULT_USER_TEMPLATE),
    ]
)


class PairwiseComparisonEvaluator(BaseEvaluator):
    """Pairwise comparison evaluator.

    Evaluates the quality of a response vs. a "reference" response given a question by
    having an LLM judge which response is better.

    Outputs whether the `response` given is better than the `reference` response.

    Args:
        service_context (Optional[ServiceContext]):
            The service context to use for evaluation.
        eval_template (Optional[Union[str, BasePromptTemplate]]):
            The template to use for evaluation.
        enforce_consensus (bool): Whether to enforce consensus (consistency if we
            flip the order of the answers). Defaults to True.

    """

    def __init__(
        self,
        service_context: Optional[ServiceContext] = None,
        eval_template: Optional[Union[BasePromptTemplate, str]] = None,
        enforce_consensus: bool = True,
    ) -> None:
        self._service_context = service_context or ServiceContext.from_defaults()

        self._eval_template: BasePromptTemplate
        if isinstance(eval_template, str):
            self._eval_template = PromptTemplate(eval_template)
        else:
            self._eval_template = eval_template or DEFAULT_EVAL_TEMPLATE

        self._enforce_consensus = enforce_consensus

    async def _get_eval_result(
        self,
        query: str,
        response: str,
        reference: str,
    ) -> EvaluationResult:
        """Get evaluation result."""
        eval_response = await self._service_context.llm_predictor.apredict(
            prompt=self._eval_template,
            query=query,
            generated_answer=response,
            reference_answer=reference,
        )

        eval_decision, eval_reason = eval_response.split("\n")

        # Extract from response
        if "2" in eval_decision.lower():
            passing: Optional[bool] = True
            score = 1.0
        elif "1" in eval_decision.lower():
            passing = False
            score = 0.0
        else:
            passing = None
            score = 0.5

        return EvaluationResult(
            query=query,
            response=response,
            passing=passing,
            score=score,
            feedback=eval_response,
        )

    async def _resolve_results(
        self,
        eval_result: EvaluationResult,
        flipped_eval_result: EvaluationResult,
    ) -> EvaluationResult:
        """Resolve eval results from evaluation + flipped evaluation."""
        if eval_result.score == flipped_eval_result.score:
            if eval_result.score == 0 or eval_result.score == 1:
                return EvaluationResult(
                    query=eval_result.query,
                    response=eval_result.response,
                    passing=None,
                    score=0.5,
                    feedback="It is not clear which answer is better.",
                )
            else:
                # Both are 0.5, return original eval_result
                return eval_result
        elif eval_result.score == 0.5:
            # in this case, flipped_eval_result.score is either 0 or 1
            # TODO: seems messy to re-flip flipped_eval_result, keep original for now
            return eval_result
        else:
            # in this case, eval_result.score is either 0 or 1
            return eval_result

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

        eval_result = await self._get_eval_result(query, response, reference)
        if self._enforce_consensus:
            # Flip the order of the answers and see if the answer is consistent
            # (which means that the score should flip from 0 to 1 and vice-versa)
            # if not, then we return a tie
            flipped_eval_result = await self._get_eval_result(
                query, reference, response
            )
            eval_result = await self._resolve_results(eval_result, flipped_eval_result)

        return eval_result
