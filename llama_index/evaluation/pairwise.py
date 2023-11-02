"""Pairwise evaluation."""

from enum import Enum
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
from llama_index.prompts.mixin import PromptDictType

DEFAULT_SYSTEM_TEMPLATE = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two "
    "AI question-answering assistants to the user question perhaps with added reference which "
    "are displayed below. You should choose the assistant that "
    "follows the user’s instructions and answers the user’s question better using the provided "
    "context. Your evaluation "
    "should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, "
    "and level of detail of their responses. Begin your evaluation by comparing the two "
    "responses and provide a short explanation. Avoid any position biases and ensure that the "
    "order in which the responses were presented does not influence your decision. Do not allow "
    "the length of the responses to influence your evaluation. Do not favor certain names of "
    "the assistants. Be as objective as possible. After providing your explanation, output your "
    "final verdict by strictly following this format: '[[A]]' if assistant A is better, '[[B]]' "
    "if assistant B is better, and '[[C]]' for a tie.\n"
)

DEFAULT_USER_TEMPLATE = (
    "[User Question]\n"
    "{query}"
    "\n\n"
    "[The Start of Reference]\n"
    "{reference}\n"
    "[The End of Reference]"
    "\n\n"
    "[The Start of Assistant A’s Answer]\n"
    "{answer_1}\n"
    "[The End of Assistant A’s Answer]"
    "\n\n"
    "[The Start of Assistant B’s Answer]\n"
    "{answer_2}\n"
    "[The End of Assistant B’s Answer]"
)

DEFAULT_EVAL_TEMPLATE = ChatPromptTemplate(
    message_templates=[
        ChatMessage(role=MessageRole.SYSTEM, content=DEFAULT_SYSTEM_TEMPLATE),
        ChatMessage(role=MessageRole.USER, content=DEFAULT_USER_TEMPLATE),
    ]
)


class EvaluationSource(str, Enum):
    """To distinguish between flipped or original."""

    ORIGINAL = "original"
    FLIPPED = "flipped"
    NEITHER = "neither"


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

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {
            "eval_template": self._eval_template,
        }

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "eval_template" in prompts:
            self._eval_template = prompts["eval_template"]

    async def _get_eval_result(
        self,
        query: str,
        response: str,
        second_response: str,
        reference: Optional[str],
    ) -> EvaluationResult:
        """Get evaluation result."""
        eval_response = await self._service_context.llm_predictor.apredict(
            prompt=self._eval_template,
            query=query,
            answer_1=response,
            answer_2=second_response,
            reference=reference or "",
        )

        # Extract from response
        if "[[A]]" in eval_response:
            passing: Optional[bool] = True
            score = 1.0
        elif "[[B]]" in eval_response:
            passing = False
            score = 0.0
        elif "[[C]]" in eval_response:
            passing = None
            score = 0.5
        else:
            raise ValueError("Unable to parse response")

        return EvaluationResult(
            query=query,
            response=eval_response,
            passing=passing,
            score=score,
            feedback=eval_response,
        )

    async def _resolve_results(
        self,
        eval_result: EvaluationResult,
        flipped_eval_result: EvaluationResult,
    ) -> EvaluationResult:
        """Resolve eval results from evaluation + flipped evaluation.

        Args:
            eval_result (EvaluationResult): Result when answer_1 is shown first
            flipped_eval_result (EvaluationResult): Result when answer_2 is shown first

        Returns:
            EvaluationResult: The final evaluation result
        """
        # add pairwise_source to eval_result and flipped_eval_result
        eval_result.pairwise_source = EvaluationSource.ORIGINAL
        flipped_eval_result.pairwise_source = EvaluationSource.FLIPPED

        # count the votes for each of the 2 answers
        votes_1 = 0.0
        votes_2 = 0.0
        if eval_result.score is not None and flipped_eval_result.score is not None:
            votes_1 = eval_result.score + (1 - flipped_eval_result.score)
            votes_2 = (1 - eval_result.score) + flipped_eval_result.score

        if votes_1 + votes_2 != 2:  # each round, the judge can give a total of 1 vote
            raise ValueError("Impossible score results. Total amount of votes is 2.")

        # get the judges (original and flipped) who voted for answer_1
        voters_1 = [eval_result] * (eval_result.score == 1.0) + [
            flipped_eval_result
        ] * (flipped_eval_result.score == 0.0)

        # get the judges (original and flipped) who voted for answer_2
        voters_2 = [eval_result] * (eval_result.score == 0.0) + [
            flipped_eval_result
        ] * (flipped_eval_result.score == 1.0)

        if votes_1 > votes_2:
            return voters_1[0]  # return any voter for answer_1
        elif votes_2 > votes_1:
            return voters_2[0]  # return any vote for answer_2
        else:
            if (
                eval_result.score == 0.5
            ):  # votes_1 == votes_2 can only happen if both are 1.0 (so actual tie)
                # doesn't matter which one we return here
                return eval_result
            else:  # Inconclusive case!
                return EvaluationResult(
                    query=eval_result.query,
                    response="",
                    passing=None,
                    score=0.5,
                    feedback="",
                    pairwise_source=EvaluationSource.NEITHER,
                )

    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        second_response: Optional[str] = None,
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        del kwargs  # Unused
        del contexts  # Unused

        if (
            query is None
            or response is None
            or second_response is None
            or reference is None
        ):
            print(query, response, second_response, reference, flush=True)
            raise ValueError(
                "query, response, second_response, and reference must be provided"
            )

        eval_result = await self._get_eval_result(
            query, response, second_response, reference
        )
        if self._enforce_consensus:
            # Flip the order of the answers and see if the answer is consistent
            # (which means that the score should flip from 0 to 1 and vice-versa)
            # if not, then we return a tie
            flipped_eval_result = await self._get_eval_result(
                query, second_response, response, reference
            )
            resolved_eval_result = await self._resolve_results(
                eval_result, flipped_eval_result
            )

        return resolved_eval_result
