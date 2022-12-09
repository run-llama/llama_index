"""Query for GPTKeywordTableIndex."""
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional

from gpt_index.indices.data_structs import KeywordTable
from gpt_index.indices.keyword_table.utils import (
    extract_keywords_given_response,
    rake_extract_keywords,
    simple_extract_keywords,
)
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.utils import truncate_text
from gpt_index.prompts.base import Prompt
from gpt_index.prompts.default_prompts import (
    DEFAULT_KEYWORD_EXTRACT_TEMPLATE,
    DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE,
    DEFAULT_REFINE_PROMPT,
    DEFAULT_TEXT_QA_PROMPT,
)

DQKET = DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE


class BaseGPTKeywordTableQuery(BaseGPTIndexQuery[KeywordTable]):
    """Base GPT Keyword Table Index Query."""

    def __init__(
        self,
        index_struct: KeywordTable,
        keyword_extract_template: Prompt = DEFAULT_KEYWORD_EXTRACT_TEMPLATE,
        query_keyword_extract_template: Optional[Prompt] = DQKET,
        refine_template: Prompt = DEFAULT_REFINE_PROMPT,
        text_qa_template: Prompt = DEFAULT_TEXT_QA_PROMPT,
        max_keywords_per_query: int = 10,
        num_chunks_per_query: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(index_struct=index_struct, **kwargs)
        self.max_keywords_per_query = max_keywords_per_query
        self.num_chunks_per_query = num_chunks_per_query
        self.keyword_extract_template = keyword_extract_template
        if query_keyword_extract_template is None:
            self.query_keyword_extract_template = keyword_extract_template
        else:
            self.query_keyword_extract_template = query_keyword_extract_template
        self.refine_template = refine_template
        self.text_qa_template = text_qa_template

    @abstractmethod
    def _get_keywords(self, query_str: str, verbose: bool = False) -> List[str]:
        """Extract keywords."""

    def query(self, query_str: str, verbose: bool = False) -> str:
        """Answer a query."""
        print(f"> Starting query: {query_str}")
        keywords = self._get_keywords(query_str, verbose=verbose)
        print(f"query keywords: {keywords}")

        # go through text chunks in order of most matching keywords
        chunk_indices_count: Dict[int, int] = defaultdict(int)
        keywords = [k for k in keywords if k in self.index_struct.keywords]
        print(f"Extracted keywords: {keywords}")
        for k in keywords:
            for text_chunk_idx in self.index_struct.table[k]:
                chunk_indices_count[text_chunk_idx] += 1
        sorted_chunk_indices = sorted(
            list(chunk_indices_count.keys()),
            key=lambda x: chunk_indices_count[x],
            reverse=True,
        )
        sorted_chunk_indices = sorted_chunk_indices[: self.num_chunks_per_query]
        result_response = None
        for text_chunk_idx in sorted_chunk_indices:
            node = self.index_struct.text_chunks[text_chunk_idx]
            fmt_text_chunk = truncate_text(node.get_text(), 50)
            print(f"> Querying with idx: {text_chunk_idx}: {fmt_text_chunk}")
            result_response = self._query_node(
                query_str,
                node,
                text_qa_template=self.text_qa_template,
                refine_template=self.refine_template,
                response=result_response,
                verbose=verbose,
            )
        return result_response or "Empty response"


class GPTKeywordTableGPTQuery(BaseGPTKeywordTableQuery):
    """GPT Keyword Table Index Query.

    Extracts keywords using GPT.

    """

    def _get_keywords(self, query_str: str, verbose: bool = False) -> List[str]:
        """Extract keywords."""
        response, _ = self._llm_predictor.predict(
            self.query_keyword_extract_template,
            max_keywords=self.max_keywords_per_query,
            question=query_str,
        )
        keywords = extract_keywords_given_response(response)
        return list(keywords)


class GPTKeywordTableSimpleQuery(BaseGPTKeywordTableQuery):
    """GPT Keyword Table Index Simple Query.

    Extracts keywords using Simple keyword extractor.

    """

    def _get_keywords(self, query_str: str, verbose: bool = False) -> List[str]:
        """Extract keywords."""
        return list(
            simple_extract_keywords(query_str, max_keywords=self.max_keywords_per_query)
        )


class GPTKeywordTableRAKEQuery(BaseGPTKeywordTableQuery):
    """GPT Keyword Table Index RAKE Query.

    Extracts keywords using RAKE keyword extractor.

    """

    def _get_keywords(self, query_str: str, verbose: bool = False) -> List[str]:
        """Extract keywords."""
        return list(
            rake_extract_keywords(query_str, max_keywords=self.max_keywords_per_query)
        )
