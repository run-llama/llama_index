"""Dataset generation from documents"""
from __future__ import annotations

from typing import List, Optional
import re

from gpt_index import (
    Document,
    GPTListIndex,
    QuestionAnswerPrompt,
    ServiceContext,
    SimpleDirectoryReader,
    LLMPredictor,
)

from langchain.chat_models import ChatOpenAI
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter

DEFAULT_QUESTION_GENERATION_PROMPT = """Context information is below.\n"
"\n---------------------\n{context_str}\n---------------------\n"
"Given the context information and not prior knowledge.\n"
"generate only questions based on the below query.\n"
"{query_str}\n"
"""


class DatasetGenerator:
    """Generate dataset (question/ question-answer pairs) \
    based on the given documents.

    NOTE: this is a beta feature, subject to change!

    Args:
        data_folder: Path to documents folder,
        model_name: "gpt-3.5-turbo" or "gpt-4",
        num_questions_per_chunk: number of question to be \
        generated per chunk. Each document is chunked of size 512 words.
        text_question_template: Question generation template.
    """

    def __init__(
        self,
        data_folder: Optional[str],
        model_name: str = "gpt-3.5-turbo",
        num_questions_per_chunk: int = 10,
        text_question_template: Optional[QuestionAnswerPrompt] = None,
        question_gen_query: Optional[str] = None,
    ) -> None:
        """Init params."""
        self.documents = SimpleDirectoryReader(data_folder).load_data()
        self.model_name = model_name
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
        self.document_chunks = self.create_document_chunks()

    def create_document_chunks(self) -> List[List[str]]:
        """
        Creates chunks for each document.
        """
        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)

        document_chunks = [
            text_splitter.split_text(document.text) for document in self.documents
        ]

        return document_chunks

    def _document_question_generator(self, chunks: List[str]) -> List[str]:
        questions = []

        for chunk in chunks:
            index = GPTListIndex.from_documents([Document(chunk)])

            llm_predictor = LLMPredictor(
                llm=ChatOpenAI(temperature=0, model_name=self.model_name)
            )
            service_context = ServiceContext.from_defaults(
                llm_predictor=llm_predictor, chunk_size_limit=3000
            )

            response = index.query(
                self.question_gen_query,
                service_context=service_context,
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

    def generate_questions(self) -> List[List[str]]:
        """
        Generates questions for each document.
        """

        questions = [
            self._document_question_generator(chunks) for chunks in self.document_chunks
        ]

        return questions
