"""List index.

A simple data structure where GPT Index iterates through document chunks
in sequence in order to answer a given query.

"""

import json
from typing import Any, Optional, Sequence
from abc import abstractmethod

from gpt_index.constants import MAX_CHUNK_OVERLAP, MAX_CHUNK_SIZE, NUM_OUTPUTS
from gpt_index.embeddings.openai import EMBED_MAX_TOKEN_LIMIT
from gpt_index.indices.base import (
    DEFAULT_MODE,
    EMBEDDING_MODE,
    BaseGPTIndex,
    BaseGPTIndexQuery,
)
from gpt_index.indices.data_structs import IndexList
from gpt_index.indices.list.embedding_query import GPTListIndexEmbeddingQuery
from gpt_index.indices.list.query import BaseGPTListIndexQuery, GPTListIndexQuery
from gpt_index.indices.utils import get_chunk_size_given_prompt, truncate_text
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.prompts.base import Prompt
from gpt_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
from gpt_index.schema import BaseDocument


# This query is used to summarize the contents of the index.
GENERATE_TEXT_QUERY = (
    "What is a concise summary of this document?"
)


class GPTListIndex(BaseGPTIndex[IndexList]):
    """GPT List Index."""

    def __init__(
        self,
        documents: Optional[Sequence[BaseDocument]] = None,
        index_struct: Optional[IndexList] = None,
        text_qa_template: Prompt = DEFAULT_TEXT_QA_PROMPT,
        llm_predictor: Optional[LLMPredictor] = None,
    ) -> None:
        """Initialize params."""
        self.text_qa_template = text_qa_template
        # we need to figure out the max length of refine_template or text_qa_template
        # to find the minimum chunk size.
        empty_qa = self.text_qa_template.format(context_str="", query_str="")

        # TODO: make embedding_limit not hardcoded.
        # To do this, we would need to include the embedding_limit in the
        # embed_model, and include that for every index.
        chunk_size = get_chunk_size_given_prompt(
            empty_qa,
            MAX_CHUNK_SIZE,
            1,
            NUM_OUTPUTS,
            embedding_limit=EMBED_MAX_TOKEN_LIMIT,
        )
        self.text_splitter = TokenTextSplitter(
            separator=" ",
            chunk_size=chunk_size,
            chunk_overlap=MAX_CHUNK_OVERLAP,
        )
        super().__init__(
            documents=documents, index_struct=index_struct, llm_predictor=llm_predictor
        )

    def _add_document_to_index(
        self, index_struct: IndexList, document: BaseDocument
    ) -> None:
        """Add document to index."""
        text_chunks = self.text_splitter.split_text(document.text)
        for _, text_chunk in enumerate(text_chunks):
            fmt_text_chunk = truncate_text(text_chunk, 50)
            print(f"> Adding chunk: {fmt_text_chunk}")
            index_struct.add_text(text_chunk)

    def build_index_from_documents(
        self, documents: Sequence[BaseDocument]
    ) -> IndexList:
        """Build the index from documents."""
        index_struct = IndexList()
        for d in documents:
            self._add_document_to_index(index_struct, d)
        return index_struct

    def _mode_to_query(self, mode: str, *query_args: Any, **query_kwargs: Any) -> BaseGPTIndexQuery:
        if mode == DEFAULT_MODE:
            if "text_qa_template" not in query_kwargs:
                query_kwargs["text_qa_template"] = self.text_qa_template
            query: BaseGPTListIndexQuery = GPTListIndexQuery(
                self.index_struct, **query_kwargs
            )
        elif mode == EMBEDDING_MODE:
            query = GPTListIndexEmbeddingQuery(self.index_struct, **query_kwargs)
        else:
            raise ValueError(f"Invalid query mode: {mode}.")
        return query

    def insert(self, document: BaseDocument, **insert_kwargs: Any) -> None:
        """Insert a document."""
        text_chunks = self.text_splitter.split_text(document.text)
        for _, text_chunk in enumerate(text_chunks):
            fmt_text_chunk = truncate_text(text_chunk, 50)
            print(f"> Adding chunk: {fmt_text_chunk}")
            self._index_struct.add_text(text_chunk)

    def delete(self, document: BaseDocument) -> None:
        """Delete a document."""
        raise NotImplementedError("Delete not implemented for list index.")

    @classmethod
    def load_from_disk(cls, save_path: str, **kwargs: Any) -> "GPTListIndex":
        """Load from disk."""
        with open(save_path, "r") as f:
            return cls(index_struct=IndexList.from_dict(json.load(f)), **kwargs)
