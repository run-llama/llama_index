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

    def build_index_from_documents(self, documents: List[Document]) -> KeywordTable:
        """Build the index from documents.

        Simply tokenize the text, excluding stopwords.

        """
        # do simple concatenation
        text_data = "\n".join([d.text for d in documents])

        index_struct = KeywordTable(table={})

        text_chunks = self.text_splitter.split_text(text_data)
        for i, text_chunk in enumerate(text_chunks):
            keywords = rake_extract_keywords(
                text_chunk, max_keywords=self.max_keywords_per_chunk
            )

            fmt_text_chunk = truncate_text(text_chunk, 50)
            text_chunk_id = index_struct.add_text(list(keywords), text_chunk)
            print(
                f"> Processing chunk {i} of {len(text_chunks)}, id {text_chunk_id}: "
                f"{fmt_text_chunk}"
            )
            print(f"> Keywords: {keywords}")
        return index_struct
