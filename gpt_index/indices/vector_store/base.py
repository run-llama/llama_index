"""Base vector store index.

An index that that is built on top of an existing vector store.

"""

from typing import Any, Optional, Sequence

from abc import abstractmethod

import numpy as np

from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.indices.base import DOCUMENTS_INPUT, BaseGPTIndex
from gpt_index.indices.data_structs import IndexDict
from gpt_index.indices.query.schema import QueryMode
from gpt_index.indices.utils import truncate_text
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
from gpt_index.prompts.prompts import QuestionAnswerPrompt
from gpt_index.schema import BaseDocument


class BaseGPTVectorStoreIndex(BaseGPTIndex[IndexDict]):
    """Base GPT Vector Store Index.

    Args:
        text_qa_template (Optional[QuestionAnswerPrompt]): A Question-Answer Prompt
            (see :ref:`Prompt-Templates`).
        embed_model (Optional[OpenAIEmbedding]): Embedding model to use for
            embedding similarity.
    """

    index_struct_cls = IndexDict

    def __init__(
        self,
        documents: Optional[Sequence[DOCUMENTS_INPUT]] = None,
        index_struct: Optional[IndexDict] = None,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        llm_predictor: Optional[LLMPredictor] = None,
        embed_model: Optional[OpenAIEmbedding] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self.text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT
        self._embed_model = embed_model or OpenAIEmbedding()
        super().__init__(
            documents=documents,
            index_struct=index_struct,
            llm_predictor=llm_predictor,
            **kwargs,
        )
        self._text_splitter = self._prompt_helper.get_text_splitter_given_prompt(
            self.text_qa_template, 1
        )

    @abstractmethod
    def _add_document_to_index(
        self,
        index_struct: IndexDict,
        document: BaseDocument,
        text_splitter: TokenTextSplitter,
    ) -> None:
        """Add document to index."""

    def _build_index_from_documents(
        self, documents: Sequence[BaseDocument], verbose: bool = False
    ) -> IndexDict:
        """Build index from documents."""
        text_splitter = self._prompt_helper.get_text_splitter_given_prompt(
            self.text_qa_template, 1
        )
        index_struct = IndexDict()
        for d in documents:
            self._add_document_to_index(index_struct, d, text_splitter)
        return index_struct

    def _insert(self, document: BaseDocument, **insert_kwargs: Any) -> None:
        """Insert a document."""
        self._add_document_to_index(self._index_struct, document, self._text_splitter)

    def delete(self, document: BaseDocument) -> None:
        """Delete a document."""
        raise NotImplementedError("Delete not implemented for Faiss index.")

