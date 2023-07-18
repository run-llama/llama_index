"""Dataset generation from documents"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple


from llama_index import (
    Document,
    ListIndex,
    QuestionAnswerPrompt,
    ServiceContext,
)
from llama_index.llms.openai import OpenAI
from llama_index.schema import BaseNode, NodeWithScore, MetadataMode
from llama_index.indices.postprocessor.node import KeywordNodePostprocessor

DEFAULT_QUESTION_GENERATION_PROMPT = """Context information is below.\n"
"\n---------------------\n{context_str}\n---------------------\n"
"Given the context information and not prior knowledge.\n"
"generate only questions based on the below query.\n"
"{query_str}\n"
"""


def _get_default_service_context() -> ServiceContext:
    """Get default service context."""
    llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
    service_context = ServiceContext.from_defaults(llm=llm, chunk_size_limit=3000)
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
        nodes: List[BaseNode],
        service_context: Optional[ServiceContext] = None,
        num_questions_per_chunk: int = 10,
        text_question_template: Optional[QuestionAnswerPrompt] = None,
        question_gen_query: Optional[str] = None,
        required_keywords: Optional[List[str]] = None,
        exclude_keywords: Optional[List[str]] = None,
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
        self.qna_gen_query = (
            self.question_gen_query
            + " Begin a question with `Q:` and an answer with `A:`."
            " Each question and answer must be on a new line."
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
        required_keywords: Optional[List[str]] = None,
        exclude_keywords: Optional[List[str]] = None,
    ) -> "DatasetGenerator":
        """Generate dataset from documents."""
        if service_context is None:
            service_context = _get_default_service_context()
        nodes = service_context.node_parser.get_nodes_from_documents(documents)

        # use node postprocessor to filter nodes
        required_keywords = required_keywords or []
        exclude_keywords = exclude_keywords or []
        node_postprocessor = KeywordNodePostprocessor(
            service_context=service_context,
            required_keywords=required_keywords,
            exclude_keywords=exclude_keywords,
        )
        node_with_scores = [NodeWithScore(node=node) for node in nodes]
        node_with_scores = node_postprocessor.postprocess_nodes(node_with_scores)
        nodes = [node_with_score.node for node_with_score in node_with_scores]

        return cls(
            nodes=nodes,
            service_context=service_context,
            num_questions_per_chunk=num_questions_per_chunk,
            text_question_template=text_question_template,
            question_gen_query=question_gen_query,
        )

    def _node_qna_generator(
        self, nodes: List[BaseNode], num: Optional[int] = None, qna_mode=False
    ) -> List[str]:
        """Node question generator."""
        output_strings: List[str] = []

        if num is not None:
            num = num * 2 if qna_mode else num
        for node in nodes:
            if num is not None and len(output_strings) >= num:
                break
            index = ListIndex.from_documents(
                [
                    Document(
                        text=node.get_content(metadata_mode=MetadataMode.NONE),
                        metadata=node.metadata,
                    )
                ]
            )

            query_engine = index.as_query_engine(
                service_context=self.service_context,
                text_qa_template=self.text_question_template,
                use_async=True,
            )
            response = query_engine.query(
                self.qna_gen_query if qna_mode else self.question_gen_query,
            )

            result = str(response).strip().split("\n")
            cleaned_output = [
                re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
            ]
            output_strings.extend(cleaned_output)

        output_strings = [string for string in output_strings if string != ""]

        if num is not None:
            output_strings = output_strings[:num]
        return output_strings

    def generate_questions_from_nodes(self, num: Optional[int] = None) -> List[str]:
        """Generates questions for each document."""
        return self._node_qna_generator(self.nodes, num, qna_mode=False)

    def generate_qna_from_nodes(
        self, num: Optional[int] = None
    ) -> List[Tuple[str, str]]:
        """Generates questions for each document."""
        output_strings = self._node_qna_generator(self.nodes, num, qna_mode=True)
        # Validate that questions start with Q: and answers start with A:
        for i in range(0, len(output_strings), 2):
            if not output_strings[i].startswith("Q:"):
                raise ValueError(f"Question {output_strings[i]} does not start with Q:")
            if not output_strings[i + 1].startswith("A:"):
                raise ValueError(f"Answer {output_strings[i+1]} does not start with A:")
            output_strings[i] = output_strings[i][2:].strip()
            output_strings[i + 1] = output_strings[i + 1][2:].strip()
        return [
            (output_strings[i], output_strings[i + 1])
            for i in range(0, len(output_strings), 2)
        ]
