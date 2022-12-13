"""RAKE keyword-table based index.

Similar to GPTKeywordTableIndex, but uses RAKE instead of GPT.

"""

from typing import Set

from gpt_index.indices.keyword_table.base import BaseGPTKeywordTableIndex
from gpt_index.indices.keyword_table.utils import rake_extract_keywords


class GPTRAKEKeywordTableIndex(BaseGPTKeywordTableIndex):
    """GPT RAKE Keyword Table Index.

    This index uses a RAKE keyword extractor to extract keywords from the text.

    """

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""
        return rake_extract_keywords(text, max_keywords=self.max_keywords_per_chunk)
