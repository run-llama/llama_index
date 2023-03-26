"""Evaluating the responses from an index."""
from __future__ import annotations

from typing import List

from llama_index import (
    Document,
    GPTListIndex,
    QuestionAnswerPrompt,
    RefinePrompt,
    Response,
)

DEFAULT_EVAL_PROMPT = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please tell if following information \
        is present in the context information: {query_str}\n\n"
    "You need to Answer with either YES/ or NO\n"
)

DEFAULT_REFINE_PROMPT = (
    "The original question is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to either YES/NO\n"
    "answer the question."
    "If the context isn't useful, return the original answer."
)


class Evaluate:
    """Evaluate based on response from indices"""

    def __init__(self, mode: str = "context_response"):
        """Initialization for evaluation.
        Args:
            mode: Two modes available with which response should be evaluated.
                  1. context_response -> comparing context \
                    information and response.
                  2. context_query_response -> comparing context \
                    information, query and response.
        """
        self.mode = mode

    def get_context(self, response: Response) -> List[Document]:
        """Get context information from given Response object using source nodes.

        Args:
            response: Response object from an index based on the query.

        Returns:
            List of Documents of source nodes information as context information.
        """

        context = []

        for context_info in response.source_nodes:
            context.append(Document(context_info.source_text))

        return context

    def evaluate(self, response: Response) -> str:
        """Evaluate the response from an index

        Args:
            response: Response object from an index based on the query.
        Returns:
            Yes -> If answer, context information are matching
            No -> If answer, context information are not matching
        """
        answer = response.response

        context = self.get_context(response)

        index = GPTListIndex(context)

        response = ""

        if self.mode == "context_response":

            EVAL_PROMPT_TMPL = QuestionAnswerPrompt(DEFAULT_EVAL_PROMPT)

            REFINE_PROMPT_TMPL = RefinePrompt(DEFAULT_REFINE_PROMPT)

            response = index.query(
                answer,
                text_qa_template=EVAL_PROMPT_TMPL,
                refine_template=REFINE_PROMPT_TMPL,
            ).response

        return response
