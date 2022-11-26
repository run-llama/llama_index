"""RAKE keyword-table based index.

Similar to GPTKeywordTableIndex, but uses RAKE instead of GPT.

"""

from typing import List

from gpt_index.indices.data_structs import KeywordTable
from gpt_index.indices.keyword_table.base import BaseGPTKeywordTableIndex
from gpt_index.indices.keyword_table.utils import rake_extract_keywords
from gpt_index.indices.utils import truncate_text
from gpt_index.schema import Document


class GPTRAKEKeywordTableIndex(BaseGPTKeywordTableIndex):
    """GPT Index."""

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        return rake_extract_keywords(
            text, max_keywords=self.max_keywords_per_chunk
        )
