"""Keyword-table based index.

Similar to a "hash table" in concept. GPT Index first tries
to extract keywords from the source text, and stores the 
keywords as keys per item. It similarly extracts keywords
from the query text. Then, it tries to match those keywords to 
existing keywords in the table.

"""

import json
from typing import Any, Dict, List, Optional, cast

from gpt_index.constants import MAX_CHUNK_OVERLAP, MAX_CHUNK_SIZE, NUM_OUTPUTS
from gpt_index.indices.base import BaseGPTIndex
from gpt_index.indices.data_structs import KeywordTable
from gpt_index.indices.utils import (
    get_chunk_size_given_prompt,
    extract_keywords_given_response,
    truncate_text,
)
from gpt_index.langchain_helpers.chain_wrapper import openai_llm_predict
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.prompts import (
    DEFAULT_KEYWORD_EXTRACT_TEMPLATE,
    DEFAULT_REFINE_PROMPT,
    DEFAULT_TEXT_QA_PROMPT
)
from gpt_index.schema import Document


class GPTKeywordTableIndex(BaseGPTIndex[KeywordTable]):
    """GPT Index."""

    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        index_struct: Optional[KeywordTable] = None,
        keyword_extract_template: str = DEFAULT_KEYWORD_EXTRACT_TEMPLATE,
        query_keyword_extract_template: Optional[str] = DEFAULT_KEYWORD_EXTRACT_TEMPLATE,
        refine_template: str = DEFAULT_REFINE_PROMPT,
        text_qa_template: str = DEFAULT_TEXT_QA_PROMPT,
        max_keywords_per_chunk: int = 6,
        max_keywords_per_query: int = 5,
        keyword_chunk_set_max_size: int = 10,
    ) -> None:
        """Initialize params."""
        # need to set parameters before building index in base class.
        self.keyword_extract_template = keyword_extract_template
        if query_keyword_extract_template is None:
            self.query_keyword_extract_template = keyword_extract_template
        else:
            self.query_keyword_extract_template = query_keyword_extract_template
        self.refine_template = refine_template
        self.text_qa_template = text_qa_template
        self.max_keywords_per_chunk = max_keywords_per_chunk
        self.max_keywords_per_query = max_keywords_per_query
        self.keyword_chunk_set_max_size = keyword_chunk_set_max_size
        super().__init__(documents=documents, index_struct=index_struct)

    
    def _refine_response(
        self, response: str, query_str: str, text_chunk: str, verbose: bool = False
    ) -> str:
        """Refine response."""
        if verbose:
            print("> Refine context: {text_chunk}")
        empty_refine_template = self.refine_template.format(
            query_str=query_str,
            existing_answer=response,
            context_msg="",
        )
        refine_chunk_size = get_chunk_size_given_prompt(
            empty_refine_template, 
            MAX_CHUNK_SIZE, 
            1, 
            NUM_OUTPUTS
        )
        refine_text_splitter = TokenTextSplitter(
            separator=" ",
            chunk_size=refine_chunk_size,
            chunk_overlap=MAX_CHUNK_OVERLAP,
        )
        text_chunks = refine_text_splitter.split_text(text_chunk)
        for text_chunk in text_chunks:
            response = openai_llm_predict(
                self.refine_template,
                query_str=query_str,
                existing_answer=response,
                context_msg=text_chunk
            )
            print("> Refined response: {response}")
        return response

    def _give_response(
        self, query_str: str, text_chunk: str, verbose: bool = False
    ) -> str:
        """Give response given a query and a corresponding text chunk."""
        empty_text_qa_template = self.text_qa_template.format(
            query_str=query_str,
            context_str="",
        )
        qa_chunk_size = get_chunk_size_given_prompt(
            empty_text_qa_template,
            MAX_CHUNK_SIZE,
            1,
            NUM_OUTPUTS
        )
        qa_text_splitter = TokenTextSplitter(
            separator=" ",
            chunk_size=qa_chunk_size,
            chunk_overlap=MAX_CHUNK_OVERLAP,
        )
        text_chunks = qa_text_splitter.split_text(text_chunk)
        response = None
        for text_chunk in text_chunks:
            if response is None:
                response = openai_llm_predict(
                    self.text_qa_template,
                    query_str=query_str,
                    context_str=text_chunk
                )
                print(f"> Initial response: {response}")
            else:
                response = self._refine_response(response, query_str, text_chunk)
        return response

    def _query_with_keyword(
        self, 
        keyword: str, 
        query_str: str, 
        result_response: Optional[str] = None,
        verbose: bool = False
    ) -> str:
        """Query with a keyword."""
        cur_response = result_response
        text_chunks = self.index_struct.get_texts(keyword)
        text_chunks = text_chunks[:self.keyword_chunk_set_max_size]
        for text_chunk in text_chunks:
            if cur_response is None:
                cur_response = self._give_response(query_str, text_chunk, verbose=verbose)
            else:
                cur_response = self._refine_response(cur_response, query_str, text_chunk, verbose=verbose)
                
        return result_response

    def query(self, query_str: str, verbose: bool = False) -> str:
        """Answer a query."""
        print(f"> Starting query: {query_str}")
        response = openai_llm_predict(
            self.query_keyword_extract_template,
            max_keywords=self.max_keywords_per_query,
            question=query_str
        )
        keywords = extract_keywords_given_response(response, self.max_keywords_per_query)

        # refine keywords based on frequency in the index (lowest frequency first)
        keywords = [k for k in keywords if k in self.index_struct.table]
        sorted_keywords = sorted(keywords, key=lambda k: len(self.index_struct.get_texts(k)))
        # NOTE: though we prompted GPT already, this is for insurance
        sorted_keywords = sorted_keywords[:self.max_keywords_per_query]
        print(f"Extracted keywords: {sorted_keywords}")

        result_response = None
        for keyword in sorted_keywords:
            result_response = self._query_with_keyword(
                keyword, query_str, result_response=result_response, verbose=verbose
            )
        return result_response

    def build_index_from_documents(self, documents: List[Document]) -> KeywordTable:
        """Build the index from documents."""
        # do simple concatenation
        text_data = "\n".join([d.text for d in documents])

        index_struct = KeywordTable(table={})

        empty_keyword_extract_template = self.keyword_extract_template.format(
            max_keywords=self.max_keywords_per_chunk,
            text=""
        )
        chunk_size = get_chunk_size_given_prompt(
            empty_keyword_extract_template, 
            MAX_CHUNK_SIZE, 
            1, 
            NUM_OUTPUTS
        )
        self.text_splitter = TokenTextSplitter(
            separator=" ",
            chunk_size=chunk_size,
            chunk_overlap=MAX_CHUNK_OVERLAP,
        )
        text_chunks = self.text_splitter.split_text(text_data)
        for i, text_chunk in enumerate(text_chunks):
            response, formatted_query_prompt = openai_llm_predict(
                self.keyword_extract_template,
                max_keywords=self.max_keywords_per_chunk,
                text=text_chunk
            )
            keywords = extract_keywords_given_response(response, self.max_keywords_per_query)
            fmt_text_chunk = truncate_text(text_chunk, 50)
            print(f"> Processing chunk {i} of {len(text_chunks)}: {fmt_text_chunk}")
            print(f"> Keywords: {keywords}")
            for keyword in keywords:
                index_struct.add_text(keyword, text_chunk)
        return index_struct

    @classmethod
    def load_from_disk(cls, save_path: str, **kwargs: Any) -> "GPTKeywordTableIndex":
        """Load from disk."""
        with open(save_path, "r") as f:
            return cls(index_struct=KeywordTable.from_dict(json.load(f)), **kwargs)

    def save_to_disk(self, save_path: str) -> None:
        """Safe to file."""
        with open(save_path, "w") as f:
            json.dump(self.index_struct.to_dict(), f)
