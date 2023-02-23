"""Query for GPTKeywordTableIndex."""
import logging
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional

from gpt_index.data_structs.data_structs import KeywordTable, Node
from gpt_index.indices.keyword_table.utils import (
    extract_keywords_given_response,
    rake_extract_keywords,
    simple_extract_keywords,
)
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.embedding_utils import SimilarityTracker
from gpt_index.indices.query.schema import QueryBundle
from gpt_index.prompts.default_prompts import (
    DEFAULT_KEYWORD_EXTRACT_TEMPLATE,
    DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE,
)
from gpt_index.prompts.prompts import KeywordExtractPrompt, QueryKeywordExtractPrompt
from gpt_index.utils import truncate_text

DQKET = DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE


class BaseGPTKeywordTableQuery(BaseGPTIndexQuery[KeywordTable]):
    """Base GPT Keyword Table Index Query.

    Arguments are shared among subclasses.

    Args:
        keyword_extract_template (Optional[KeywordExtractPrompt]): A Keyword
            Extraction Prompt
            (see :ref:`Prompt-Templates`).
        query_keyword_extract_template (Optional[QueryKeywordExtractPrompt]): A Query
            Keyword Extraction
            Prompt (see :ref:`Prompt-Templates`).
        refine_template (Optional[RefinePrompt]): A Refinement Prompt
            (see :ref:`Prompt-Templates`).
        text_qa_template (Optional[QuestionAnswerPrompt]): A Question Answering Prompt
            (see :ref:`Prompt-Templates`).
        max_keywords_per_query (int): Maximum number of keywords to extract from query.
        num_chunks_per_query (int): Maximum number of text chunks to query.

    """

    def __init__(
        self,
        index_struct: KeywordTable,
        keyword_extract_template: Optional[KeywordExtractPrompt] = None,
        query_keyword_extract_template: Optional[QueryKeywordExtractPrompt] = None,
        max_keywords_per_query: int = 10,
        num_chunks_per_query: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(index_struct=index_struct, **kwargs)
        self.max_keywords_per_query = max_keywords_per_query
        self.num_chunks_per_query = num_chunks_per_query
        self.keyword_extract_template = (
            keyword_extract_template or DEFAULT_KEYWORD_EXTRACT_TEMPLATE
        )
        self.query_keyword_extract_template = query_keyword_extract_template or DQKET

    @abstractmethod
    def _get_keywords(self, query_str: str) -> List[str]:
        """Extract keywords."""

    def _get_nodes_for_response(
        self,
        query_bundle: QueryBundle,
        similarity_tracker: Optional[SimilarityTracker] = None,
    ) -> List[Node]:
        """Get nodes for response."""
        logging.info(f"> Starting query: {query_bundle.query_str}")
        keywords = self._get_keywords(query_bundle.query_str)
        logging.info(f"query keywords: {keywords}")

        # go through text chunks in order of most matching keywords
        chunk_indices_count: Dict[int, int] = defaultdict(int)
        keywords = [k for k in keywords if k in self.index_struct.keywords]
        logging.info(f"> Extracted keywords: {keywords}")
        for k in keywords:
            for text_chunk_idx in self.index_struct.table[k]:
                chunk_indices_count[text_chunk_idx] += 1
        sorted_chunk_indices = sorted(
            list(chunk_indices_count.keys()),
            key=lambda x: chunk_indices_count[x],
            reverse=True,
        )
        sorted_chunk_indices = sorted_chunk_indices[: self.num_chunks_per_query]
        sorted_nodes = [
            self.index_struct.text_chunks[idx] for idx in sorted_chunk_indices
        ]
        # filter sorted nodes
        sorted_nodes = [node for node in sorted_nodes if self._should_use_node(node)]

        if logging.getLogger(__name__).getEffectiveLevel() == logging.DEBUG:
            for chunk_idx, node in zip(sorted_chunk_indices, sorted_nodes):
                logging.debug(
                    f"> Querying with idx: {chunk_idx}: "
                    f"{truncate_text(node.get_text(), 50)}"
                )

        return sorted_nodes


class GPTKeywordTableGPTQuery(BaseGPTKeywordTableQuery):
    """GPT Keyword Table Index Query.

    Extracts keywords using GPT. Set when `mode="default"` in `query` method of
    `GPTKeywordTableIndex`.

    .. code-block:: python

        response = index.query("<query_str>", mode="default")

    See BaseGPTKeywordTableQuery for arguments.

    """

    def _get_keywords(self, query_str: str) -> List[str]:
        """Extract keywords."""
        response, _ = self._llm_predictor.predict(
            self.query_keyword_extract_template,
            max_keywords=self.max_keywords_per_query,
            question=query_str,
        )
        keywords = extract_keywords_given_response(response, start_token="KEYWORDS:")
        return list(keywords)


class GPTKeywordTableSimpleQuery(BaseGPTKeywordTableQuery):
    """GPT Keyword Table Index Simple Query.

    Extracts keywords using simple regex-based keyword extractor.
    Set when `mode="simple"` in `query` method of `GPTKeywordTableIndex`.

    .. code-block:: python

        response = index.query("<query_str>", mode="simple")

    See BaseGPTKeywordTableQuery for arguments.

    """

    def _get_keywords(self, query_str: str) -> List[str]:
        """Extract keywords."""
        return list(
            simple_extract_keywords(query_str, max_keywords=self.max_keywords_per_query)
        )


class GPTKeywordTableRAKEQuery(BaseGPTKeywordTableQuery):
    """GPT Keyword Table Index RAKE Query.

    Extracts keywords using RAKE keyword extractor.
    Set when `mode="rake"` in `query` method of `GPTKeywordTableIndex`.

    .. code-block:: python

        response = index.query("<query_str>", mode="rake")

    See BaseGPTKeywordTableQuery for arguments.

    """

    def _get_keywords(self, query_str: str) -> List[str]:
        """Extract keywords."""
        return list(
            rake_extract_keywords(query_str, max_keywords=self.max_keywords_per_query)
        )
