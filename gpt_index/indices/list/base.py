"""List index.

A simple data structure where GPT Index iterates through document chunks
in sequence in order to answer a given query.

"""

import json
from typing import Any, List, Optional

from gpt_index.constants import MAX_CHUNK_OVERLAP, MAX_CHUNK_SIZE, NUM_OUTPUTS
from gpt_index.indices.base import BaseGPTIndex
from gpt_index.indices.data_structs import IndexList
from gpt_index.indices.response_utils import give_response, refine_response
from gpt_index.indices.utils import get_chunk_size_given_prompt, truncate_text
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.prompts import DEFAULT_REFINE_PROMPT, DEFAULT_TEXT_QA_PROMPT
from gpt_index.schema import Document


class GPTListIndex(BaseGPTIndex[IndexList]):
    """GPT List Index."""

    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        index_struct: Optional[IndexList] = None,
        refine_template: str = DEFAULT_REFINE_PROMPT,
        text_qa_template: str = DEFAULT_TEXT_QA_PROMPT,
    ) -> None:
        """Initialize params."""
        self.refine_template = refine_template
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

    def query(self, query_str: str, verbose: bool = False) -> str:
        """Answer a query."""
        print(f"> Starting query: {query_str}")
        response = None
        for node in self.index_struct.nodes:
            fmt_text_chunk = truncate_text(node.text, 50)
            if verbose:
                print(f"> Searching in chunk: {fmt_text_chunk}")

            # TODO: abstract create and refine procedure
            if response is None:
                response = give_response(
                    query_str,
                    node.text,
                    text_qa_template=self.text_qa_template,
                    refine_template=self.refine_template,
                    verbose=verbose,
                )
            else:
                response = refine_response(
                    response,
                    query_str,
                    node.text,
                    refine_template=self.refine_template,
                    verbose=verbose,
                )
            if verbose:
                print(f"> Response: {response}")
        return response or ""

    @classmethod
    def load_from_disk(cls, save_path: str, **kwargs: Any) -> "GPTListIndex":
        """Load from disk."""
        with open(save_path, "r") as f:
            return cls(index_struct=IndexList.from_dict(json.load(f)), **kwargs)

    def save_to_disk(self, save_path: str) -> None:
        """Safe to file."""
        with open(save_path, "w") as f:
            json.dump(self.index_struct.to_dict(), f)
