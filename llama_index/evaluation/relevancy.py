"""Relevancy evaluation."""
from __future__ import annotations

import asyncio
import re
from typing import Any, Sequence

from llama_index import ServiceContext
from llama_index.evaluation.base import BaseEvaluator, EvaluationResult
from llama_index.indices import SummaryIndex
from llama_index.prompts import BasePromptTemplate, PromptTemplate
from llama_index.prompts.mixin import PromptDictType
from llama_index.schema import Document

DEFAULT_EVAL_TEMPLATE = PromptTemplate(
    "Your task is to evaluate if the response for the query "
    "is in line with the context information provided.\n"
    "You have two options to answer: Either 'YES' or 'NO'.\n"
    "Additionally, provide a brief reasoning for your answer.\n"
    "Answer 'YES' if the response for the query "
    "is in line with context information, and explain why. "
    "Answer 'NO' if it is not, and explain why not.\n"
    "Some examples are provided below.\n\n"
    "Example 1:\n"
    "Question: What is the capital of France?\n"
    "Response: Paris.\n"
    "Context: France is a country in Europe. Its capital city is Paris.\n"
    "Answer: YES\n"
    "Reasoning: The context confirms that Paris is the capital of France.\n\n"
    "Example 2:\n"
    "Question: What is the boiling point of water?\n"
    "Response: 100 degrees Celsius.\n"
    "Context: Water is essential for life, and about 71% of the Earth's surface is water-covered.\n"
    "Answer: NO\n"
    "Reasoning: The context does not provide information related to the boiling point of water, hence it cannot be used to verify the response.\n\n"
    "{query_str}\n"
    "Context: {context_str}\n"
)


DEFAULT_REFINE_TEMPLATE = PromptTemplate(
    "We want to understand if the following query and response is"
    "in line with the context information: \n {query_str}\n"
    "We have provided an existing YES/NO answer with reasoning: \n {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "If the existing answer was already YES, still answer YES. "
    "If the information is present in the new context, answer YES. "
    "Otherwise answer NO.\n"
    "Make sure to update the reasoning if needed.\n"
)


class RelevancyEvaluator(BaseEvaluator):
    """Relenvancy evaluator.

    Evaluates the relevancy of retrieved contexts and response to a query.
    This evaluator considers the query string, retrieved contexts, and response string.

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
        service_context: ServiceContext | None = None,
        raise_error: bool = False,
        eval_template: str | BasePromptTemplate | None = None,
        refine_template: str | BasePromptTemplate | None = None,
    ) -> None:
        """Init params."""
        self._service_context = service_context or ServiceContext.from_defaults()
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
        """Evaluate whether the contexts and response are relevant to the query."""
        del kwargs  # Unused

        if query is None or contexts is None or response is None:
            raise ValueError("query, contexts, and response must be provided")

        docs = [Document(text=context) for context in contexts]
        index = SummaryIndex.from_documents(docs, service_context=self._service_context)

        query_response = f"Question: {query}\nResponse: {response}"

        await asyncio.sleep(sleep_time_in_seconds)

        query_engine = index.as_query_engine(
            text_qa_template=self._eval_template,
            refine_template=self._refine_template,
        )
        response_obj = await query_engine.aquery(query_response)

        raw_response_txt = str(response_obj)

        if "yes" in raw_response_txt.lower():
            passing = True
        else:
            if self._raise_error:
                raise ValueError("The response is invalid")
            passing = False

        reasoning_pattern = re.compile(r"^Reasoning: (.+)$", re.MULTILINE)
        reasoning_match = reasoning_pattern.search(raw_response_txt)
        reasoning = reasoning_match.group(1) if reasoning_match else None

        return EvaluationResult(
            query=query,
            response=response,
            passing=passing,
            score=1.0 if passing else 0.0,
            feedback=reasoning,
        )


QueryResponseEvaluator = RelevancyEvaluator
