"""Faithfulness evaluation."""

from __future__ import annotations

import asyncio
from typing import Any, Optional, Sequence, Union

from llama_index.core.evaluation.base import BaseEvaluator, EvaluationResult
from llama_index.core.indices import SummaryIndex
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import BasePromptTemplate, PromptTemplate
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.schema import Document
from llama_index.core.settings import Settings

DEFAULT_EVAL_TEMPLATE = PromptTemplate(
    "Please tell if a given piece of information "
    "is supported by the context.\n"
    "You need to answer with either YES or NO.\n"
    "Answer YES if any of the context supports the information, even "
    "if most of the context is unrelated. "
    "Some examples are provided below. \n\n"
    "Information: Apple pie is generally double-crusted.\n"
    "Context: An apple pie is a fruit pie in which the principal filling "
    "ingredient is apples. \n"
    "Apple pie is often served with whipped cream, ice cream "
    "('apple pie à la mode'), custard or cheddar cheese.\n"
    "It is generally double-crusted, with pastry both above "
    "and below the filling; the upper crust may be solid or "
    "latticed (woven of crosswise strips).\n"
    "Answer: YES\n"
    "Information: Apple pies tastes bad.\n"
    "Context: An apple pie is a fruit pie in which the principal filling "
    "ingredient is apples. \n"
    "Apple pie is often served with whipped cream, ice cream "
    "('apple pie à la mode'), custard or cheddar cheese.\n"
    "It is generally double-crusted, with pastry both above "
    "and below the filling; the upper crust may be solid or "
    "latticed (woven of crosswise strips).\n"
    "Answer: NO\n"
    "Information: {query_str}\n"
    "Context: {context_str}\n"
    "Answer: "
)

DEFAULT_REFINE_TEMPLATE = PromptTemplate(
    "We want to understand if the following information is present "
    "in the context information: {query_str}\n"
    "We have provided an existing YES/NO answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "If the existing answer was already YES, still answer YES. "
    "If the information is present in the new context, answer YES. "
    "Otherwise answer NO.\n"
)

LLAMA3_8B_EVAL_TEMPLATE = PromptTemplate(
    """Please tell if a given piece of information is supported by the context.
You need to answer with either YES or NO.
Answer YES if **any part** of the context supports the information, even if most of the context is unrelated.
Answer NO if the context does not support the information at all.
Be sure to read all provided context segments carefully before making your decision.

Some examples are provided below:

Example 1:
Information: The Eiffel Tower is located in Paris.
Context: The Eiffel Tower, a symbol of French culture, stands prominently in the city of Paris.
Answer: YES

Example 2:
Information: Bananas are a type of berry.
Context: Bananas are a popular fruit enjoyed worldwide and are rich in potassium.
Answer: NO

Example 3:
Information: Cats are reptiles.
Context: Cats are domesticated felines known for their agility and companionship.
Answer: NO

Example 4:
Information: Amazon started as an online bookstore.
Context: Amazon initially launched as an online store for books but has since expanded into a global e-commerce giant
offering various products and services.
Answer: YES

Information: {query}
Context: {reference_contexts}
Answer:"""
)

TEMPLATES_CATALOG = {"llama3:8b": LLAMA3_8B_EVAL_TEMPLATE}


class FaithfulnessEvaluator(BaseEvaluator):
    """
    Faithfulness evaluator.

    Evaluates whether a response is faithful to the contexts
    (i.e. whether the response is supported by the contexts or hallucinated.)

    This evaluator only considers the response string and the list of context strings.

    Args:
        raise_error(bool): Whether to raise an error when the response is invalid.
            Defaults to False.
        eval_template(Optional[Union[str, BasePromptTemplate]]):
            The template to use for evaluation.
        refine_template(Optional[Union[str, BasePromptTemplate]]):
            The template to use for refining the evaluation.
    """

    def __init__(
        self,
        llm: Optional[LLM] = None,
        raise_error: bool = False,
        eval_template: Optional[Union[str, BasePromptTemplate]] = None,
        refine_template: Optional[Union[str, BasePromptTemplate]] = None,
    ) -> None:
        """Init params."""
        self._llm = llm or Settings.llm
        self._raise_error = raise_error

        self._eval_template: BasePromptTemplate
        if isinstance(eval_template, str):
            self._eval_template = PromptTemplate(eval_template)
        if isinstance(eval_template, BasePromptTemplate):
            self._eval_template = eval_template
        else:
            model_name = self._llm.metadata.model_name
            self._eval_template = TEMPLATES_CATALOG.get(
                model_name, DEFAULT_EVAL_TEMPLATE
            )

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
        """Evaluate whether the response is faithful to the contexts."""
        del kwargs  # Unused

        await asyncio.sleep(sleep_time_in_seconds)

        if contexts is None or response is None:
            raise ValueError("contexts and response must be provided")

        docs = [Document(text=context) for context in contexts]
        index = SummaryIndex.from_documents(docs)

        query_engine = index.as_query_engine(
            llm=self._llm,
            text_qa_template=self._eval_template,
            refine_template=self._refine_template,
        )
        response_obj = await query_engine.aquery(response)

        raw_response_txt = str(response_obj)

        if "yes" in raw_response_txt.lower():
            passing = True
        else:
            passing = False
            if self._raise_error:
                raise ValueError("The response is invalid")

        return EvaluationResult(
            query=query,
            response=response,
            contexts=contexts,
            passing=passing,
            score=1.0 if passing else 0.0,
            feedback=raw_response_txt,
        )


# legacy: backward compatibility
ResponseEvaluator = FaithfulnessEvaluator
