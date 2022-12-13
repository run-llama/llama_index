"""Simple keyword-table based index.

Similar to GPTKeywordTableIndex, but uses a simpler keyword extraction
technique that doesn't involve GPT - just uses regex.

"""

from typing import Set

from gpt_index.indices.keyword_table.base import BaseGPTKeywordTableIndex
from gpt_index.indices.keyword_table.utils import simple_extract_keywords
from gpt_index.prompts.default_prompts import DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE

DQKET = DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE


class GPTSimpleKeywordTableIndex(BaseGPTKeywordTableIndex):
    """GPT Simple Keyword Table Index.

    This index uses a simple regex extractor to extract keywords from the text.

    """

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""
        return simple_extract_keywords(text, self.max_keywords_per_chunk)
