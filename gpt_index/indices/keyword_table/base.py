"""Keyword-table based index.

Similar to a "hash table" in concept. GPT Index first tries
to extract keywords from the source text, and stores the
keywords as keys per item. It similarly extracts keywords
from the query text. Then, it tries to match those keywords to
existing keywords in the table.

"""

from abc import abstractmethod
from typing import Any, Optional, Sequence, Set

from gpt_index.indices.base import DOCUMENTS_INPUT, BaseGPTIndex
from gpt_index.indices.data_structs import KeywordTable
from gpt_index.indices.keyword_table.utils import extract_keywords_given_response
from gpt_index.indices.utils import truncate_text
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.prompts.default_prompts import (
    DEFAULT_KEYWORD_EXTRACT_TEMPLATE,
    DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE,
)
from gpt_index.prompts.prompts import KeywordExtractPrompt
from gpt_index.schema import BaseDocument

DQKET = DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE


class BaseGPTKeywordTableIndex(BaseGPTIndex[KeywordTable]):
    """GPT Keyword Table Index.

    This index extracts keywords from the text, and maps each
    keyword to the node(s) that it corresponds to. In this sense it mimicks a
    "hash table". During index construction, the keyword table is constructed
    by extracting keywords from each node and creating an internal mapping.

    During query time, the keywords are extracted from the query text, and these
    keywords are used to index into the keyword table. The retrieved nodes
    are then used to answer the query.

    Args:
        keyword_extract_template (Optional[KeywordExtractPrompt]): A Keyword
            Extraction Prompt
            (see :ref:`Prompt-Templates`).

    """

    index_struct_cls = KeywordTable

    def __init__(
        self,
        documents: Optional[Sequence[DOCUMENTS_INPUT]] = None,
        index_struct: Optional[KeywordTable] = None,
        keyword_extract_template: Optional[KeywordExtractPrompt] = None,
        max_keywords_per_chunk: int = 10,
        llm_predictor: Optional[LLMPredictor] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        # need to set parameters before building index in base class.
        self.keyword_extract_template = (
            keyword_extract_template or DEFAULT_KEYWORD_EXTRACT_TEMPLATE
        )
        self.max_keywords_per_chunk = max_keywords_per_chunk
        super().__init__(
            documents=documents,
            index_struct=index_struct,
            llm_predictor=llm_predictor,
            **kwargs,
        )
        self._text_splitter = self._prompt_helper.get_text_splitter_given_prompt(
            self.keyword_extract_template, 1
        )

    @abstractmethod
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""

    def _add_document_to_index(
        self,
        index_struct: KeywordTable,
        document: BaseDocument,
        text_splitter: TokenTextSplitter,
    ) -> None:
        """Add document to index."""
        text_chunks = text_splitter.split_text(document.get_text())
        for i, text_chunk in enumerate(text_chunks):
            keywords = self._extract_keywords(text_chunk)
            fmt_text_chunk = truncate_text(text_chunk, 50)
            text_chunk_id = index_struct.add_text(
                list(keywords), text_chunk, document.get_doc_id()
            )
            print(
                f"> Processing chunk {i} of {len(text_chunks)}, id {text_chunk_id}: "
                f"{fmt_text_chunk}"
            )
            print(f"> Keywords: {keywords}")

    def _build_index_from_documents(
        self, documents: Sequence[BaseDocument], verbose: bool = False
    ) -> KeywordTable:
        """Build the index from documents."""
        text_splitter = self._prompt_helper.get_text_splitter_given_prompt(
            self.keyword_extract_template, 1
        )
        # do simple concatenation
        index_struct = KeywordTable(table={})
        for d in documents:
            self._add_document_to_index(index_struct, d, text_splitter)

        return index_struct

    def _insert(self, document: BaseDocument, **insert_kwargs: Any) -> None:
        """Insert a document."""
        text_chunks = self._text_splitter.split_text(document.get_text())
        for i, text_chunk in enumerate(text_chunks):
            keywords = self._extract_keywords(text_chunk)
            fmt_text_chunk = truncate_text(text_chunk, 50)
            text_chunk_id = self._index_struct.add_text(
                list(keywords), text_chunk, document.get_doc_id()
            )
            print(
                f"> Processing chunk {i} of {len(text_chunks)}, id {text_chunk_id}: "
                f"{fmt_text_chunk}"
            )
            print(f"> Keywords: {keywords}")

    def delete(self, document: BaseDocument) -> None:
        """Delete a document."""
        raise NotImplementedError("Delete not implemented for keyword table index.")


class GPTKeywordTableIndex(BaseGPTKeywordTableIndex):
    """GPT Keyword Table Index.

    This index uses a GPT model to extract keywords from the text.

    """

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""
        response, _ = self._llm_predictor.predict(
            self.keyword_extract_template,
            max_keywords=self.max_keywords_per_chunk,
            text=text,
        )
        keywords = extract_keywords_given_response(response)
        return keywords
