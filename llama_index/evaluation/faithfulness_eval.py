"""Evaluating the responses from an index."""
from __future__ import annotations

from typing import List, Optional

from llama_index.evaluation.base import BaseEvaluator
from llama_index.evaluation.utils import get_context
from llama_index.indices.base import ServiceContext
from llama_index.indices.list.base import SummaryIndex
from llama_index.prompts import PromptTemplate
from llama_index.response.schema import Response

DEFAULT_EVAL_PROMPT = (
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

DEFAULT_REFINE_PROMPT = (
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


class FaithfulnessEvaluator(BaseEvaluator):
    """Evaluate the faithfulness of 


    """

    def __init__(
        self,
        service_context: Optional[ServiceContext] = None,
        raise_error: bool = False,
        eval_template: Optional[str] = None,
        refine_template: Optional[str] = None,
    ) -> None:
        """Init params."""
        self._service_context = service_context or ServiceContext.from_defaults()
        self._raise_error = raise_error
        self._eval_template = eval_template or DEFAULT_EVAL_PROMPT
        self._refine_template = refine_template or DEFAULT_REFINE_PROMPT

    def evaluate(self, response: Response) -> str:
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
        answer = str(response)

        context = get_context(response)
        index = SummaryIndex.from_documents(
            context, service_context=self.service_context
        )
        response_txt = ""

        EVAL_PROMPT_TMPL = PromptTemplate(DEFAULT_EVAL_PROMPT)
        REFINE_PROMPT_TMPL = PromptTemplate(DEFAULT_REFINE_PROMPT)

        query_engine = index.as_query_engine(
            text_qa_template=EVAL_PROMPT_TMPL,
            refine_template=REFINE_PROMPT_TMPL,
        )
        response_obj = query_engine.query(answer)

        raw_response_txt = str(response_obj)

        if "yes" in raw_response_txt.lower():
            response_txt = "YES"
        else:
            response_txt = "NO"
            if self.raise_error:
                raise ValueError("The response is invalid")

        return response_txt

    def evaluate_source_nodes(self, response: Response) -> List[str]:
        """Function to evaluate if each source node contains the answer \
            by comparing the response, and context information (source node).

        Args:
            response: Response object from an index based on the query.
        Returns:
            List of Yes/ No which can be used to know which source node contains \
                answer.
            Yes -> If response and context information are matching.
            No -> If response and context information are not matching.
        """
        answer = str(response)

        context_list = self.get_context(response)

        response_texts = []

        for context in context_list:
            index = SummaryIndex.from_documents(
                [context], service_context=self.service_context
            )
            response_txt = ""

            EVAL_PROMPT_TMPL = PromptTemplate(DEFAULT_EVAL_PROMPT)
            REFINE_PROMPT_TMPL = PromptTemplate(DEFAULT_REFINE_PROMPT)

            query_engine = index.as_query_engine(
                text_qa_template=EVAL_PROMPT_TMPL,
                refine_template=REFINE_PROMPT_TMPL,
            )
            response_obj = query_engine.query(answer)
            raw_response_txt = str(response_obj)

            if "yes" in raw_response_txt.lower():
                response_txt = "YES"
            else:
                response_txt = "NO"
                if self.raise_error:
                    raise ValueError("The response is invalid")

            response_texts.append(response_txt)

        return response_texts
    


ResponseEvaluator = FaithfulnessEvaluator