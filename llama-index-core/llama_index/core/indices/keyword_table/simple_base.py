"""
Simple keyword-table based index.

Similar to KeywordTableIndex, but uses a simpler keyword extraction
technique that doesn't involve GPT - just uses regex.

"""

from typing import Any, Set, Union

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.keyword_table.base import (
    BaseKeywordTableIndex,
    KeywordTableRetrieverMode,
)
from llama_index.core.indices.keyword_table.utils import simple_extract_keywords
from llama_index.core.prompts.default_prompts import (
    DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE,
)

DQKET = DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE


class SimpleKeywordTableIndex(BaseKeywordTableIndex):
    """
    Simple Keyword Table Index.

    This index uses a simple regex extractor to extract keywords from the text.

    """

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""
        return simple_extract_keywords(text, self.max_keywords_per_chunk)

    def as_retriever(
        self,
        retriever_mode: Union[
            str, KeywordTableRetrieverMode
        ] = KeywordTableRetrieverMode.SIMPLE,
        **kwargs: Any,
    ) -> BaseRetriever:
        return super().as_retriever(retriever_mode=retriever_mode, **kwargs)


# legacy
GPTSimpleKeywordTableIndex = SimpleKeywordTableIndex
