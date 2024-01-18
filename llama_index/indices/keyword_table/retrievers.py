"""Query for KeywordTableIndex."""
import logging
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional

from llama_index.callbacks.base import CallbackManager
from llama_index.core.base_retriever import BaseRetriever
from llama_index.indices.keyword_table.base import BaseKeywordTableIndex
from llama_index.indices.keyword_table.utils import (
    extract_keywords_given_response,
    rake_extract_keywords,
    simple_extract_keywords,
)
from llama_index.prompts import BasePromptTemplate
from llama_index.prompts.default_prompts import (
    DEFAULT_KEYWORD_EXTRACT_TEMPLATE,
    DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE,
)
from llama_index.schema import NodeWithScore, QueryBundle
from llama_index.utils import truncate_text

DQKET = DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE

logger = logging.getLogger(__name__)


class BaseKeywordTableRetriever(BaseRetriever):
    """Base Keyword Table Retriever.

    Arguments are shared among subclasses.

    Args:
        keyword_extract_template (Optional[BasePromptTemplate]): A Keyword
            Extraction Prompt
            (see :ref:`Prompt-Templates`).
        query_keyword_extract_template (Optional[BasePromptTemplate]): A Query
            Keyword Extraction
            Prompt (see :ref:`Prompt-Templates`).
        refine_template (Optional[BasePromptTemplate]): A Refinement Prompt
            (see :ref:`Prompt-Templates`).
        text_qa_template (Optional[BasePromptTemplate]): A Question Answering Prompt
            (see :ref:`Prompt-Templates`).
        max_keywords_per_query (int): Maximum number of keywords to extract from query.
        num_chunks_per_query (int): Maximum number of text chunks to query.

    """

    def __init__(
        self,
        index: BaseKeywordTableIndex,
        keyword_extract_template: Optional[BasePromptTemplate] = None,
        query_keyword_extract_template: Optional[BasePromptTemplate] = None,
        max_keywords_per_query: int = 10,
        num_chunks_per_query: int = 10,
        callback_manager: Optional[CallbackManager] = None,
        object_map: Optional[dict] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._index = index
        self._index_struct = index.index_struct
        self._docstore = index.docstore
        self._service_context = index.service_context

        self.max_keywords_per_query = max_keywords_per_query
        self.num_chunks_per_query = num_chunks_per_query
        self.keyword_extract_template = (
            keyword_extract_template or DEFAULT_KEYWORD_EXTRACT_TEMPLATE
        )
        self.query_keyword_extract_template = query_keyword_extract_template or DQKET
        super().__init__(
            callback_manager=callback_manager,
            object_map=object_map,
            verbose=verbose,
        )

    @abstractmethod
    def _get_keywords(self, query_str: str) -> List[str]:
        """Extract keywords."""

    def _retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        """Get nodes for response."""
        logger.info(f"> Starting query: {query_bundle.query_str}")
        keywords = self._get_keywords(query_bundle.query_str)
        logger.info(f"query keywords: {keywords}")

        # go through text chunks in order of most matching keywords
        chunk_indices_count: Dict[str, int] = defaultdict(int)
        keywords = [k for k in keywords if k in self._index_struct.keywords]
        logger.info(f"> Extracted keywords: {keywords}")
        for k in keywords:
            for node_id in self._index_struct.table[k]:
                chunk_indices_count[node_id] += 1
        sorted_chunk_indices = sorted(
            chunk_indices_count.keys(),
            key=lambda x: chunk_indices_count[x],
            reverse=True,
        )
        sorted_chunk_indices = sorted_chunk_indices[: self.num_chunks_per_query]
        sorted_nodes = self._docstore.get_nodes(sorted_chunk_indices)

        if logging.getLogger(__name__).getEffectiveLevel() == logging.DEBUG:
            for chunk_idx, node in zip(sorted_chunk_indices, sorted_nodes):
                logger.debug(
                    f"> Querying with idx: {chunk_idx}: "
                    f"{truncate_text(node.get_content(), 50)}"
                )
        return [NodeWithScore(node=node) for node in sorted_nodes]


class KeywordTableGPTRetriever(BaseKeywordTableRetriever):
    """Keyword Table Index GPT Retriever.

    Extracts keywords using GPT. Set when using `retriever_mode="default"`.

    See BaseGPTKeywordTableQuery for arguments.

    """

    def _get_keywords(self, query_str: str) -> List[str]:
        """Extract keywords."""
        response = self._service_context.llm.predict(
            self.query_keyword_extract_template,
            max_keywords=self.max_keywords_per_query,
            question=query_str,
        )
        keywords = extract_keywords_given_response(response, start_token="KEYWORDS:")
        return list(keywords)


class KeywordTableSimpleRetriever(BaseKeywordTableRetriever):
    """Keyword Table Index Simple Retriever.

    Extracts keywords using simple regex-based keyword extractor.
    Set when `retriever_mode="simple"`.

    See BaseGPTKeywordTableQuery for arguments.

    """

    def _get_keywords(self, query_str: str) -> List[str]:
        """Extract keywords."""
        return list(
            simple_extract_keywords(query_str, max_keywords=self.max_keywords_per_query)
        )


class KeywordTableRAKERetriever(BaseKeywordTableRetriever):
    """Keyword Table Index RAKE Retriever.

    Extracts keywords using RAKE keyword extractor.
    Set when `retriever_mode="rake"`.

    See BaseGPTKeywordTableQuery for arguments.

    """

    def _get_keywords(self, query_str: str) -> List[str]:
        """Extract keywords."""
        return list(
            rake_extract_keywords(query_str, max_keywords=self.max_keywords_per_query)
        )
