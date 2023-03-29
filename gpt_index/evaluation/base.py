"""Evaluating the responses from an index."""
from __future__ import annotations

from typing import List, Optional

from gpt_index import (
    Document,
    GPTListIndex,
    QuestionAnswerPrompt,
    RefinePrompt,
    Response,
    ServiceContext,
)

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


class ResponseEvaluator:
    """Evaluate based on response from indices.

    NOTE: this is a beta feature, subject to change!

    Args:
        mode (str): Mode with which the response should be evaluated.
            1. context_response -> comparing context \
            information and response.
            2. others coming soon!
    
    """

    def __init__(
        self,
        mode: str = "context_response",
        service_context: Optional[ServiceContext] = None,
    ) -> None:
        """Init params."""
        self.mode = mode
        self.service_context = service_context or ServiceContext.from_defaults()

    def get_context(self, response: Response) -> List[Document]:
        """Get context information from given Response object using source nodes.

        Args:
            response (Response): Response object from an index based on the query.

        Returns:
            List of Documents of source nodes information as context information.
        """

        context = []

        for context_info in response.source_nodes:
            context.append(Document(context_info.source_text))

        return context

    def evaluate(self, response: Response) -> str:
        """Evaluate the response from an index.

        Args:
            response: Response object from an index based on the query.
        Returns:
            Yes -> If answer, context information are matching
            No -> If answer, context information are not matching
        """
        answer = str(response)

        context = self.get_context(response)
        index = GPTListIndex.from_documents(
            context, service_context=self.service_context
        )
        response_txt: str = ""

        if self.mode == "context_response":
            EVAL_PROMPT_TMPL = QuestionAnswerPrompt(DEFAULT_EVAL_PROMPT)
            REFINE_PROMPT_TMPL = RefinePrompt(DEFAULT_REFINE_PROMPT)

            response_obj = index.query(
                answer,
                text_qa_template=EVAL_PROMPT_TMPL,
                refine_template=REFINE_PROMPT_TMPL,
            )
            response_txt = str(response_obj)
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented.")

        return response_txt
