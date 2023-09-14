"""Relevancy evaluation."""
from __future__ import annotations

from typing import Any, Optional, Sequence

from llama_index.evaluation.base import BaseEvaluator, EvaluationResult
from llama_index.indices.base import ServiceContext
from llama_index.indices.list.base import SummaryIndex
from llama_index.prompts import PromptTemplate
from llama_index.schema import Document

QUERY_RESPONSE_EVAL_PROMPT = (
    "Your task is to evaluate if the response for the query \
    is in line with the context information provided.\n"
    "You have two options to answer. Either YES/ NO.\n"
    "Answer - YES, if the response for the query \
    is in line with context information otherwise NO.\n"
    "Query and Response: \n {query_str}\n"
    "Context: \n {context_str}\n"
    "Answer: "
)

QUERY_RESPONSE_REFINE_PROMPT = (
    "We want to understand if the following query and response is"
    "in line with the context information: \n {query_str}\n"
    "We have provided an existing YES/NO answer: \n {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "If the existing answer was already YES, still answer YES. "
    "If the information is present in the new context, answer YES. "
    "Otherwise answer NO.\n"
)


class RelevancyEvaluator(BaseEvaluator):
    """Evaluate based on query and response from indices.

    NOTE: this is a beta feature, subject to change!

    Args:
        service_context (Optional[ServiceContext]): ServiceContext object

    """

    def __init__(
        self,
        service_context: Optional[ServiceContext] = None,
        raise_error: bool = False,
    ) -> None:
        """Init params."""
        self._service_context = service_context or ServiceContext.from_defaults()
        self._raise_error = raise_error

    def evaluate(
        self,
        query: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        response: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate the response from an index.

        Args:
            query: Query for which response is generated from index.
            response: Response object from an index based on the query.
        Returns:
            EvaluationResult object with passing boolean and feedback "YES" or "NO".
        """
        if query is None or contexts is None or response is None:
            raise ValueError("query, contexts, and response must be provided")

        docs = [Document(text=context) for context in contexts]
        index = SummaryIndex.from_documents(docs, service_context=self._service_context)

        QUERY_RESPONSE_EVAL_PROMPT_TMPL = PromptTemplate(QUERY_RESPONSE_EVAL_PROMPT)
        QUERY_RESPONSE_REFINE_PROMPT_TMPL = PromptTemplate(QUERY_RESPONSE_REFINE_PROMPT)

        query_response = f"Question: {query}\nResponse: {response}"

        query_engine = index.as_query_engine(
            text_qa_template=QUERY_RESPONSE_EVAL_PROMPT_TMPL,
            refine_template=QUERY_RESPONSE_REFINE_PROMPT_TMPL,
        )
        response_obj = query_engine.query(query_response)

        raw_response_txt = str(response_obj)

        if "yes" in raw_response_txt.lower():
            return EvaluationResult(query, response, True, "YES")
        else:
            if self._raise_error:
                raise ValueError("The response is invalid")
            return EvaluationResult(query, response, False, "NO")

QueryResponseEvaluator = RelevancyEvaluator