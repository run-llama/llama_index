"""Dataset generation from documents."""

from __future__ import annotations

import re
import warnings
from typing import List, Sequence, Optional

from llama_index.core import Document, SummaryIndex
from llama_index.core.async_utils import DEFAULT_NUM_WORKERS, run_jobs, asyncio_run
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.ingestion import run_transformations
from llama_index.core.llama_dataset import (
    CreatedBy,
    CreatedByType,
    LabelledRagDataExample,
    LabelledRagDataset,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.postprocessor.node import KeywordNodePostprocessor
from llama_index.core.prompts.base import BasePromptTemplate, PromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
from llama_index.core.prompts.mixin import (
    PromptDictType,
    PromptMixin,
    PromptMixinType,
)
from llama_index.core.schema import (
    BaseNode,
    MetadataMode,
    NodeWithScore,
    TransformComponent,
)
from llama_index.core.settings import Settings


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
        num_questions_per_chunk: number of question to be \
        generated per chunk. Each document is chunked of size 512 words.
        text_question_template: Question generation template.
        question_gen_query: Question generation query.

    """

    def __init__(
        self,
        nodes: List[BaseNode],
        llm: Optional[LLM] = None,
        num_questions_per_chunk: int = 3,
        text_question_template: Optional[BasePromptTemplate] = None,
        text_qa_template: Optional[BasePromptTemplate] = None,
        question_gen_query: Optional[str] = None,
        metadata_mode: MetadataMode = MetadataMode.NONE,
        show_progress: bool = False,
        workers: int = DEFAULT_NUM_WORKERS,
    ) -> None:
        """Init params."""
        self._llm = llm or Settings.llm
        self.num_questions_per_chunk = num_questions_per_chunk
        self.text_question_template = text_question_template or PromptTemplate(
            DEFAULT_QUESTION_GENERATION_PROMPT
        )
        self.text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT
        self.question_gen_query = (
            question_gen_query
            or f"You are a Teacher/Professor. Your task is to setup {num_questions_per_chunk} questions for an upcoming quiz/examination. The questions should be diverse in nature across the document. Restrict the questions to the context information provided."
        )
        self.nodes = nodes
        self._metadata_mode = metadata_mode
        self._show_progress = show_progress
        self._workers = workers

    @classmethod
    def from_documents(
        cls,
        documents: Sequence[Document],
        llm: Optional[LLM] = None,
        transformations: Optional[List[TransformComponent]] = None,
        num_questions_per_chunk: int = 3,
        text_question_template: Optional[BasePromptTemplate] = None,
        text_qa_template: Optional[BasePromptTemplate] = None,
        question_gen_query: Optional[str] = None,
        required_keywords: Optional[List[str]] = None,
        exclude_keywords: Optional[List[str]] = None,
        show_progress: bool = False,
        workers: int = DEFAULT_NUM_WORKERS,
    ) -> RagDatasetGenerator:
        """Generate dataset from documents."""
        llm = llm or Settings.llm
        transformations = transformations or Settings.transformations

        nodes = run_transformations(
            documents, transformations, show_progress=show_progress
        )

        # use node postprocessor to filter nodes
        required_keywords = required_keywords or []
        exclude_keywords = exclude_keywords or []
        node_postprocessor = KeywordNodePostprocessor(
            required_keywords=required_keywords,
            exclude_keywords=exclude_keywords,
        )
        node_with_scores = [NodeWithScore(node=node) for node in nodes]
        node_with_scores = node_postprocessor.postprocess_nodes(node_with_scores)
        nodes = [node_with_score.node for node_with_score in node_with_scores]

        return cls(
            nodes=nodes,
            llm=llm,
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
                        metadata=node.metadata,  # type: ignore
                        excluded_llm_metadata_keys=node.excluded_llm_metadata_keys,
                        excluded_embed_metadata_keys=node.excluded_embed_metadata_keys,
                        relationships=node.relationships,
                    )
                ],
            )

            query_engine = index.as_query_engine(
                llm=self._llm,
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
            ][: self.num_questions_per_chunk]

            num_questions_generated = len(cleaned_questions)
            if num_questions_generated < self.num_questions_per_chunk:
                warnings.warn(
                    f"Fewer questions generated ({num_questions_generated}) "
                    f"than requested ({self.num_questions_per_chunk})."
                )

            index = summary_indices[idx]
            reference_context = nodes[idx].get_content(metadata_mode=MetadataMode.NONE)
            model_name = self._llm.metadata.model_name
            created_by = CreatedBy(type=CreatedByType.AI, model_name=model_name)
            if labelled:
                index = summary_indices[idx]
                qr_tasks = []
                for query in cleaned_questions:
                    # build summary index off of node (i.e. context)
                    qa_query_engine = index.as_query_engine(
                        llm=self._llm,
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
                    example = LabelledRagDataExample(
                        query=question,
                        reference_answer=str(answer_response),
                        reference_contexts=[reference_context],
                        reference_answer_by=created_by,
                        query_by=created_by,
                    )
                    examples.append(example)
            else:
                for query in cleaned_questions:
                    example = LabelledRagDataExample(
                        query=query,
                        reference_answer="",
                        reference_contexts=[reference_context],
                        reference_answer_by=None,
                        query_by=created_by,
                    )
                    examples.append(example)

        # split train/test
        return LabelledRagDataset(examples=examples)

    async def agenerate_questions_from_nodes(self) -> LabelledRagDataset:
        """Generates questions but not the reference answers."""
        return await self._agenerate_dataset(self.nodes, labelled=False)

    async def agenerate_dataset_from_nodes(self) -> LabelledRagDataset:
        """Generates questions for each document."""
        return await self._agenerate_dataset(self.nodes, labelled=True)

    def generate_questions_from_nodes(self) -> LabelledRagDataset:
        """Generates questions but not the reference answers."""
        return asyncio_run(self.agenerate_questions_from_nodes())

    def generate_dataset_from_nodes(self) -> LabelledRagDataset:
        """Generates questions for each document."""
        return asyncio_run(self.agenerate_dataset_from_nodes())

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
