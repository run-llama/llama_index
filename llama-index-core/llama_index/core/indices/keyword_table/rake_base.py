"""
RAKE keyword-table based index.

Similar to KeywordTableIndex, but uses RAKE instead of GPT.

"""

from typing import Any, Set, Union

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.keyword_table.base import (
    BaseKeywordTableIndex,
    KeywordTableRetrieverMode,
)
from llama_index.core.indices.keyword_table.utils import rake_extract_keywords


class RAKEKeywordTableIndex(BaseKeywordTableIndex):
    """
    RAKE Keyword Table Index.

    This index uses a RAKE keyword extractor to extract keywords from the text.

    """

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""
        return rake_extract_keywords(text, max_keywords=self.max_keywords_per_chunk)

    def as_retriever(
        self,
        retriever_mode: Union[
            str, KeywordTableRetrieverMode
        ] = KeywordTableRetrieverMode.RAKE,
        **kwargs: Any,
    ) -> BaseRetriever:
        return super().as_retriever(retriever_mode=retriever_mode, **kwargs)


# legacy
GPTRAKEKeywordTableIndex = RAKEKeywordTableIndex
