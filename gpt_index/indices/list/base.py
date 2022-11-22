"""List index.

A simple data structure where GPT Index iterates through document chunks
in sequence in order to answer a given query.

"""

import json
from typing import Any, List, Optional

from gpt_index.constants import MAX_CHUNK_OVERLAP, MAX_CHUNK_SIZE, NUM_OUTPUTS
from gpt_index.indices.base import DEFAULT_MODE, BaseGPTIndex, BaseGPTIndexQuery
from gpt_index.indices.data_structs import IndexList
from gpt_index.indices.list.query import GPTListIndexQuery
from gpt_index.indices.utils import get_chunk_size_given_prompt, truncate_text
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.prompts.base import Prompt
from gpt_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
from gpt_index.schema import Document


class GPTListIndex(BaseGPTIndex[IndexList]):
    """GPT List Index."""

    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        index_struct: Optional[IndexList] = None,
        text_qa_template: Prompt = DEFAULT_TEXT_QA_PROMPT,
    ) -> None:
        """Initialize params."""
        self.text_qa_template = text_qa_template
        # we need to figure out the max length of refine_template or text_qa_template
        # to find the minimum chunk size.
        empty_qa = self.text_qa_template.format(context_str="", query_str="")
        chunk_size = get_chunk_size_given_prompt(
            empty_qa, MAX_CHUNK_SIZE, 1, NUM_OUTPUTS
        )
        self.text_splitter = TokenTextSplitter(
            separator=" ",
            chunk_size=chunk_size,
            chunk_overlap=MAX_CHUNK_OVERLAP,
        )
        super().__init__(documents=documents, index_struct=index_struct)

    def build_index_from_documents(self, documents: List[Document]) -> IndexList:
        """Build the index from documents."""
        # do a simple concatenation
        text_data = "\n".join([d.text for d in documents])
        index_struct = IndexList()
        text_chunks = self.text_splitter.split_text(text_data)
        for _, text_chunk in enumerate(text_chunks):
            fmt_text_chunk = truncate_text(text_chunk, 50)
            print(f"> Adding chunk: {fmt_text_chunk}")
            index_struct.add_text(text_chunk)
        return index_struct

    def _mode_to_query(self, mode: str, **query_kwargs: Any) -> BaseGPTIndexQuery:
        if mode == DEFAULT_MODE:
            query_kwargs.update(
                {
                    "text_qa_template": self.text_qa_template,
                }
            )
            query = GPTListIndexQuery(self.index_struct, **query_kwargs)
        else:
            raise ValueError(f"Invalid query mode: {mode}.")
        return query

    @classmethod
    def load_from_disk(cls, save_path: str, **kwargs: Any) -> "GPTListIndex":
        """Load from disk."""
        with open(save_path, "r") as f:
            return cls(index_struct=IndexList.from_dict(json.load(f)), **kwargs)

    def save_to_disk(self, save_path: str) -> None:
        """Safe to file."""
        with open(save_path, "w") as f:
            json.dump(self.index_struct.to_dict(), f)
