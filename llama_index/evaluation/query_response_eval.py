"""Evaluating the responses from an index."""
from __future__ import annotations

from typing import List, Optional

from llama_index.evaluation.base import BaseEvaluator, Evaluation
from llama_index.evaluation.utils import get_context
from llama_index.indices.base import ServiceContext
from llama_index.indices.list.base import SummaryIndex
from llama_index.prompts import PromptTemplate
from llama_index.response.schema import Response

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


class QueryResponseEvaluator(BaseEvaluator):
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
        self.service_context = service_context or ServiceContext.from_defaults()
        self.raise_error = raise_error

    def evaluate(self, query: str, response: Response) -> str:
        """Evaluate the response from an index.

        Args:
            query: Query for which response is generated from index.
            response: Response object from an index based on the query.
        Returns:
            Yes -> If answer, context information are matching \
                    or If Query, answer and context information are matching.
            No -> If answer, context information are not matching \
                    or If Query, answer and context information are not matching.
        """
        return self.evaluate_response(query, response).feedback

    def evaluate_response(self, query: str, response: Response) -> Evaluation:
        """Evaluate the response from an index.

        Args:
            query: Query for which response is generated from index.
            response: Response object from an index based on the query.
        Returns:
            Evaluation object with passing boolean and feedback "YES" or "NO".
        """
        answer = str(response)

        context = get_context(response)
        index = SummaryIndex.from_documents(
            context, service_context=self.service_context
        )

        QUERY_RESPONSE_EVAL_PROMPT_TMPL = PromptTemplate(QUERY_RESPONSE_EVAL_PROMPT)
        QUERY_RESPONSE_REFINE_PROMPT_TMPL = PromptTemplate(QUERY_RESPONSE_REFINE_PROMPT)

        query_response = f"Question: {query}\nResponse: {answer}"

        query_engine = index.as_query_engine(
            text_qa_template=QUERY_RESPONSE_EVAL_PROMPT_TMPL,
            refine_template=QUERY_RESPONSE_REFINE_PROMPT_TMPL,
        )
        response_obj = query_engine.query(query_response)

        raw_response_txt = str(response_obj)

        if "yes" in raw_response_txt.lower():
            return Evaluation(query, response, True, "YES")
        else:
            if self.raise_error:
                raise ValueError("The response is invalid")
            return Evaluation(query, response, False, "NO")

    def evaluate_source_nodes(self, query: str, response: Response) -> List[str]:
        """Function to evaluate if each source node contains the answer \
            to a given query by comparing the query, response, \
                and context information.

        Args:
            query: Query for which response is generated from index.
            response: Response object from an index based on the query.
        Returns:
            List of Yes/ No which can be used to know which source node contains \
                answer.
            Yes -> If answer, context information are matching \
                    or If Query, answer and context information are matching \
                        for a source node.
            No -> If answer, context information are not matching \
                    or If Query, answer and context information are not matching \
                        for a source node.
        """
        answer = str(response)

        context_list = self.get_context(response)

        response_texts = []

        for context in context_list:
            index = SummaryIndex.from_documents(
                [context], service_context=self.service_context
            )
            response_txt = ""

            QUERY_RESPONSE_EVAL_PROMPT_TMPL = PromptTemplate(QUERY_RESPONSE_EVAL_PROMPT)
            QUERY_RESPONSE_REFINE_PROMPT_TMPL = PromptTemplate(
                QUERY_RESPONSE_REFINE_PROMPT
            )

            query_response = f"Question: {query}\nResponse: {answer}"

            query_engine = index.as_query_engine(
                text_qa_template=QUERY_RESPONSE_EVAL_PROMPT_TMPL,
                refine_template=QUERY_RESPONSE_REFINE_PROMPT_TMPL,
            )
            response_obj = query_engine.query(query_response)
            raw_response_txt = str(response_obj)

            if "yes" in raw_response_txt.lower():
                response_txt = "YES"
            else:
                response_txt = "NO"
                if self.raise_error:
                    raise ValueError("The response is invalid")

            response_texts.append(response_txt)

        return response_texts
