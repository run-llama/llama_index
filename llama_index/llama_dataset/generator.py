"""Dataset generation from documents."""
from __future__ import annotations

import asyncio
import re
from typing import List

from llama_index import Document, ServiceContext, SummaryIndex
from llama_index.async_utils import DEFAULT_NUM_WORKERS, run_jobs
from llama_index.core.response.schema import RESPONSE_TYPE
from llama_index.ingestion import run_transformations
from llama_index.llama_dataset import (
    CreatedBy,
    CreatedByType,
    LabelledRagDataExample,
    LabelledRagDataset,
)
from llama_index.postprocessor.node import KeywordNodePostprocessor
from llama_index.prompts.base import BasePromptTemplate, PromptTemplate
from llama_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
from llama_index.prompts.mixin import PromptDictType, PromptMixin, PromptMixinType
from llama_index.schema import BaseNode, MetadataMode, NodeWithScore

DEFAULT_QUESTION_GENERATION_PROMPT = """\
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge.
generate only questions based on the below query.
{query_str}
"""


class RagDatasetGenerator(PromptMixin):
    """Generate dataset (question/ question-answer pairs) \
    based on the given documents.

    NOTE: this is a beta feature, subject to change!

    Args:
        nodes (List[Node]): List of nodes. (Optional)
        service_context (ServiceContext): Service Context.
        num_questions_per_chunk: number of question to be \
        generated per chunk. Each document is chunked of size 512 words.
        text_question_template: Question generation template.
        question_gen_query: Question generation query.

    """

    def __init__(
        self,
        nodes: List[BaseNode],
        service_context: ServiceContext | None = None,
        num_questions_per_chunk: int = 3,
        text_question_template: BasePromptTemplate | None = None,
        text_qa_template: BasePromptTemplate | None = None,
        question_gen_query: str | None = None,
        metadata_mode: MetadataMode = MetadataMode.NONE,
        show_progress: bool = False,
        workers: int = DEFAULT_NUM_WORKERS,
    ) -> None:
        """Init params."""
        if service_context is None:
            service_context = service_context or ServiceContext.from_defaults(
                chunk_size_limit=3000
            )
        self.service_context = service_context
        self.text_question_template = text_question_template or PromptTemplate(
            DEFAULT_QUESTION_GENERATION_PROMPT
        )
        self.text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT
        self.question_gen_query = (
            question_gen_query
            or f"You are a Teacher/Professor. Your task is to setup \
                        {num_questions_per_chunk} questions for an upcoming \
                        quiz/examination. The questions should be diverse in nature \
                            across the document. Restrict the questions to the \
                                context information provided."
        )
        self.nodes = nodes
        self._metadata_mode = metadata_mode
        self._show_progress = show_progress
        self._workers = workers

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        service_context: ServiceContext | None = None,
        num_questions_per_chunk: int = 3,
        text_question_template: BasePromptTemplate | None = None,
        text_qa_template: BasePromptTemplate | None = None,
        question_gen_query: str | None = None,
        required_keywords: List[str] | None = None,
        exclude_keywords: List[str] | None = None,
        show_progress: bool = False,
        workers: int = DEFAULT_NUM_WORKERS,
    ) -> RagDatasetGenerator:
        """Generate dataset from documents."""
        if service_context is None:
            service_context = service_context or ServiceContext.from_defaults(
                chunk_size_limit=3000
            )

        nodes = run_transformations(
            documents, service_context.transformations, show_progress=show_progress
        )

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
            text_qa_template=text_qa_template,
            question_gen_query=question_gen_query,
            show_progress=show_progress,
            workers=workers,
        )

    async def _agenerate_dataset(
        self,
        nodes: List[BaseNode],
        labelled: bool = False,
    ) -> LabelledRagDataset:
        """Node question generator."""
        query_tasks = []
        examples: List[LabelledRagDataExample] = []
        summary_indices: List[SummaryIndex] = []
        for node in nodes:
            index = SummaryIndex.from_documents(
                [
                    Document(
                        text=node.get_content(metadata_mode=self._metadata_mode),
                        metadata=node.metadata,
                    )
                ],
                service_context=self.service_context,
            )

            query_engine = index.as_query_engine(
                service_context=self.service_context,
                text_qa_template=self.text_question_template,
                use_async=True,
            )
            task = query_engine.aquery(
                self.question_gen_query,
            )
            query_tasks.append(task)
            summary_indices.append(index)

        responses = await run_jobs(query_tasks, self._show_progress, self._workers)
        for idx, response in enumerate(responses):
            result = str(response).strip().split("\n")
            cleaned_questions = [
                re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
            ]
            cleaned_questions = [
                question for question in cleaned_questions if len(question) > 0
            ]
            index = summary_indices[idx]
            reference_context = nodes[idx].text

            if labelled:
                index = summary_indices[idx]
                qr_tasks = []
                for query in cleaned_questions:
                    # build summary index off of node (i.e. context)
                    qa_query_engine = index.as_query_engine(
                        service_context=self.service_context,
                        text_qa_template=self.text_qa_template,
                    )
                    qr_task = qa_query_engine.aquery(query)
                    qr_tasks.append(qr_task)
                answer_responses: List[RESPONSE_TYPE] = await run_jobs(
                    qr_tasks, self._show_progress, self._workers
                )
                for question, answer_response in zip(
                    cleaned_questions, answer_responses
                ):
                    model_name = self.service_context.llm.metadata.model_name
                    created_by = CreatedBy(type=CreatedByType.AI, model_name=model_name)
                    example = LabelledRagDataExample(
                        query=question,
                        reference_answer=str(answer_response),
                        reference_contexts=[reference_context],
                        reference_answer_by=created_by,
                        query_by=created_by,
                    )
                    examples.append(example)
            else:
                pass

        # split train/test
        return LabelledRagDataset(examples=examples)

    async def agenerate_questions_from_nodes(self) -> List[str]:
        """Generates questions for each document."""
        dataset = await self._agenerate_dataset(self.nodes, labelled=False)
        return dataset.questions

    async def agenerate_dataset_from_nodes(self) -> LabelledRagDataset:
        """Generates questions for each document."""
        return await self._agenerate_dataset(self.nodes, labelled=True)

    def generate_questions_from_nodes(self) -> List[str]:
        """Generates questions for each document."""
        return asyncio.run(self.agenerate_questions_from_nodes())

    def generate_dataset_from_nodes(self) -> LabelledRagDataset:
        """Generates questions for each document."""
        return asyncio.run(self.agenerate_dataset_from_nodes())

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {
            "text_question_template": self.text_question_template,
            "text_qa_template": self.text_qa_template,
        }

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt modules."""
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "text_question_template" in prompts:
            self.text_question_template = prompts["text_question_template"]
        if "text_qa_template" in prompts:
            self.text_qa_template = prompts["text_qa_template"]
