"""Relevancy evaluation."""

from __future__ import annotations

import asyncio
import re
from typing import Any, Callable, Optional, Sequence, Tuple

from llama_index.core.evaluation.base import BaseEvaluator, EvaluationResult
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import BasePromptTemplate, PromptTemplate
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.settings import Settings

DEFAULT_EVAL_TEMPLATE = PromptTemplate(
    "Your task is to evaluate if the response is relevant to the query.\n"
    "The evaluation should be performed in a step-by-step manner by answering the following questions:\n"
    "1. Does the provided response match the subject matter of the user's query?\n"
    "2. Does the provided response attempt to address the focus or perspective "
    "on the subject matter taken on by the user's query?\n"
    "Each question above is worth 1 point. Provide detailed feedback on response according to the criteria questions above  "
    "After your feedback provide a final result by strictly following this format: '[RESULT] followed by the integer number representing the total score assigned to the response'\n\n"
    "Query: \n {query}\n"
    "Response: \n {response}\n"
    "Feedback:"
)

_DEFAULT_SCORE_THRESHOLD = 2.0


def _default_parser_function(output_str: str) -> Tuple[Optional[float], Optional[str]]:
    # Pattern to match the feedback and response
    # This pattern looks for any text ending with '[RESULT]' followed by a number
    pattern = r"([\s\S]+)(?:\[RESULT\]\s*)(\d)"

    # Using regex to find all matches
    result = re.search(pattern, output_str)

    # Check if any match is found
    if result:
        # Assuming there's only one match in the text, extract feedback and response
        feedback, score = result.groups()
        score = float(score) if score is not None else score
        return score, feedback.strip()
    else:
        return None, None


class AnswerRelevancyEvaluator(BaseEvaluator):
    """
    Answer relevancy evaluator.

    Evaluates the relevancy of response to a query.
    This evaluator considers the query string and response string.

    Args:
        raise_error(Optional[bool]):
            Whether to raise an error if the response is invalid.
            Defaults to False.
        eval_template(Optional[Union[str, BasePromptTemplate]]):
            The template to use for evaluation.
        refine_template(Optional[Union[str, BasePromptTemplate]]):
            The template to use for refinement.

    """

    def __init__(
        self,
        llm: Optional[LLM] = None,
        raise_error: bool = False,
        eval_template: str | BasePromptTemplate | None = None,
        score_threshold: float = _DEFAULT_SCORE_THRESHOLD,
        parser_function: Callable[
            [str], Tuple[Optional[float], Optional[str]]
        ] = _default_parser_function,
    ) -> None:
        """Init params."""
        self._llm = llm or Settings.llm
        self._raise_error = raise_error

        self._eval_template: BasePromptTemplate
        if isinstance(eval_template, str):
            self._eval_template = PromptTemplate(eval_template)
        else:
            self._eval_template = eval_template or DEFAULT_EVAL_TEMPLATE

        self.parser_function = parser_function
        self.score_threshold = score_threshold

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {
            "eval_template": self._eval_template,
            "refine_template": self._refine_template,
        }

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "eval_template" in prompts:
            self._eval_template = prompts["eval_template"]
        if "refine_template" in prompts:
            self._refine_template = prompts["refine_template"]

    async def aevaluate(
        self,
        query: str | None = None,
        response: str | None = None,
        contexts: Sequence[str] | None = None,
        sleep_time_in_seconds: int = 0,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate whether the response is relevant to the query."""
        del kwargs  # Unused
        del contexts  # Unused

        if query is None or response is None:
            raise ValueError("query and response must be provided")

        await asyncio.sleep(sleep_time_in_seconds)

        eval_response = await self._llm.apredict(
            prompt=self._eval_template,
            query=query,
            response=response,
        )

        score, reasoning = self.parser_function(eval_response)

        invalid_result, invalid_reason = False, None
        if score is None and reasoning is None:
            if self._raise_error:
                raise ValueError("The response is invalid")
            invalid_result = True
            invalid_reason = "Unable to parse the output string."

        if score:
            score /= self.score_threshold

        return EvaluationResult(
            query=query,
            response=response,
            score=score,
            feedback=eval_response,
            invalid_result=invalid_result,
            invalid_reason=invalid_reason,
        )
