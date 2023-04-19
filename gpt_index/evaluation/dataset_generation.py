"""Dataset generation from documents"""
from __future__ import annotations

from typing import List, Optional
import re

from gpt_index import (
    Document,
    GPTListIndex,
    QuestionAnswerPrompt,
    ServiceContext,
    LLMPredictor,
)
from gpt_index.data_structs.node_v2 import Node

from langchain.chat_models import ChatOpenAI

DEFAULT_QUESTION_GENERATION_PROMPT = """Context information is below.\n"
"\n---------------------\n{context_str}\n---------------------\n"
"Given the context information and not prior knowledge.\n"
"generate only questions based on the below query.\n"
"{query_str}\n"
"""


def _get_default_service_context() -> ServiceContext:
    """Get default service context."""
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    )
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, chunk_size_limit=3000
    )
    return service_context


class DatasetGenerator:
    """Generate dataset (question/ question-answer pairs) \
    based on the given documents.

    NOTE: this is a beta feature, subject to change!

    Args:
        nodes (List[Node]): List of nodes. (Optional)
        service_context (ServiceContext): Service Context.
        num_questions_per_chunk: number of question to be \
        generated per chunk. Each document is chunked of size 512 words.
        text_question_template: Question generation template.
    """

    def __init__(
        self,
        nodes: List[Node],
        service_context: Optional[ServiceContext] = None,
        num_questions_per_chunk: int = 10,
        text_question_template: Optional[QuestionAnswerPrompt] = None,
        question_gen_query: Optional[str] = None,
    ) -> None:
        """Init params."""
        if service_context is None:
            service_context = _get_default_service_context()
        self.service_context = service_context
        self.text_question_template = text_question_template or QuestionAnswerPrompt(
            DEFAULT_QUESTION_GENERATION_PROMPT
        )
        self.question_gen_query = (
            question_gen_query
            or f"You are a Teacher/ Professor. Your task is to setup \
                        {num_questions_per_chunk} questions for an upcoming \
                        quiz/examination. The questions should be diverse in nature \
                            across the document. Restrict the questions to the \
                                context information provided."
        )
        self.nodes = nodes

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        service_context: Optional[ServiceContext] = None,
        num_questions_per_chunk: int = 10,
        text_question_template: Optional[QuestionAnswerPrompt] = None,
        question_gen_query: Optional[str] = None,
    ) -> "DatasetGenerator":
        """Generate dataset from documents."""
        if service_context is None:
            service_context = _get_default_service_context()
        nodes = service_context.node_parser.get_nodes_from_documents(documents)

        return cls(
            nodes=nodes,
            service_context=service_context,
            num_questions_per_chunk=num_questions_per_chunk,
            text_question_template=text_question_template,
            question_gen_query=question_gen_query,
        )

    def _node_question_generator(self, nodes: List[Node]) -> List[str]:
        """Node question generator."""
        questions = []

        for node in nodes:
            index = GPTListIndex.from_documents([Document(node.get_text())])

            response = index.query(
                self.question_gen_query,
                service_context=self.service_context,
                text_qa_template=self.text_question_template,
                use_async=True,
            )

            result = str(response).strip().split("\n")
            cleaned_questions = [
                re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
            ]
            questions.extend(cleaned_questions)

        questions = [question for question in questions if question != ""]

        return questions

    def generate_questions_from_nodes(self) -> List[str]:
        """Generates questions for each document."""
        return self._node_question_generator(self.nodes)
