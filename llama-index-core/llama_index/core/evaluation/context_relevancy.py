"""Relevancy evaluation."""
from __future__ import annotations

import asyncio
import re
from typing import Any, Callable, Optional, Sequence, Tuple

from llama_index.core import ServiceContext
from llama_index.core.evaluation.base import BaseEvaluator, EvaluationResult
from llama_index.core.indices import SummaryIndex
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import BasePromptTemplate, PromptTemplate
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.schema import Document
from llama_index.core.settings import Settings, llm_from_settings_or_context

DEFAULT_EVAL_TEMPLATE = PromptTemplate(
    "Your task is to evaluate if the retrieved context from the document sources are relevant to the query.\n"
    "The evaluation should be performed in a step-by-step manner by answering the following questions:\n"
    "1. Does the retrieved context match the subject matter of the user's query?\n"
    "2. Can the retrieved context be used exclusively to provide a full answer to the user's query?\n"
    "Each question above is worth 2 points, where partial marks are allowed and encouraged. Provide detailed feedback on the response "
    "according to the criteria questions previously mentioned. "
    "After your feedback provide a final result by strictly following this format: "
    "'[RESULT] followed by the float number representing the total score assigned to the response'\n\n"
    "Query: \n {query_str}\n"
    "Context: \n {context_str}\n"
    "Feedback:"
)

_DEFAULT_SCORE_THRESHOLD = 4.0

DEFAULT_REFINE_TEMPLATE = PromptTemplate(
    "We want to understand if the following query and response is"
    "in line with the context information: \n {query_str}\n"
    "We have provided an existing evaluation score: \n {existing_answer}\n"
    "We have the opportunity to refine the existing evaluation "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    f"If the existing evaluation was already {_DEFAULT_SCORE_THRESHOLD}, still answer {_DEFAULT_SCORE_THRESHOLD}. "
    f"If the information is present in the new context, answer {_DEFAULT_SCORE_THRESHOLD}. "
    "Otherwise answer {existing_answer}.\n"
)


def _default_parser_function(output_str: str) -> Tuple[Optional[float], Optional[str]]:
    # Pattern to match the feedback and response
    # This pattern looks for any text ending with '[RESULT]' followed by a number
    pattern = r"([\s\S]+)(?:\[RESULT\]\s*)([\d.]+)"

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


class ContextRelevancyEvaluator(BaseEvaluator):
    """Context relevancy evaluator.

    Evaluates the relevancy of retrieved contexts to a query.
    This evaluator considers the query string and retrieved contexts.

    Args:
        service_context(Optional[ServiceContext]):
            The service context to use for evaluation.
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
        refine_template: str | BasePromptTemplate | None = None,
        score_threshold: float = _DEFAULT_SCORE_THRESHOLD,
        parser_function: Callable[
            [str], Tuple[Optional[float], Optional[str]]
        ] = _default_parser_function,
        # deprecated
        service_context: Optional[ServiceContext] = None,
    ) -> None:
        """Init params."""
        self._llm = llm or llm_from_settings_or_context(Settings, service_context)
        self._raise_error = raise_error

        self._eval_template: BasePromptTemplate
        if isinstance(eval_template, str):
            self._eval_template = PromptTemplate(eval_template)
        else:
            self._eval_template = eval_template or DEFAULT_EVAL_TEMPLATE

        self._refine_template: BasePromptTemplate
        if isinstance(refine_template, str):
            self._refine_template = PromptTemplate(refine_template)
        else:
            self._refine_template = refine_template or DEFAULT_REFINE_TEMPLATE

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
        """Evaluate whether the contexts is relevant to the query."""
        del kwargs  # Unused
        del response  # Unused

        if query is None or contexts is None:
            raise ValueError("Both query and contexts must be provided")

        docs = [Document(text=context) for context in contexts]
        index = SummaryIndex.from_documents(docs)

        await asyncio.sleep(sleep_time_in_seconds)

        query_engine = index.as_query_engine(
            llm=self._llm,
            text_qa_template=self._eval_template,
            refine_template=self._refine_template,
        )
        response_obj = await query_engine.aquery(query)
        raw_response_txt = str(response_obj)

        score, reasoning = self.parser_function(raw_response_txt)

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
            contexts=contexts,
            score=score,
            feedback=raw_response_txt,
            invalid_result=invalid_result,
            invalid_reason=invalid_reason,
        )
