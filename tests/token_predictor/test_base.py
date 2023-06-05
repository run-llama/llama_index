"""Test token predictor."""

from typing import Any
from unittest.mock import MagicMock, patch

from langchain.llms.base import BaseLLM

from llama_index.indices.keyword_table.base import GPTKeywordTableIndex
from llama_index.indices.list.base import ListIndex
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.tree.base import TreeIndex
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.readers.schema.base import Document
from llama_index.token_counter.mock_chain_wrapper import MockLLMPredictor
from tests.mock_utils.mock_text_splitter import mock_token_splitter_newline


@patch.object(TokenTextSplitter, "split_text", side_effect=mock_token_splitter_newline)
def test_token_predictor(mock_split: Any) -> None:
    """Test token predictor."""
    # here, just assert that token predictor runs (before checking behavior)
    # TODO: mock token counting a bit more carefully
    doc_text = (
        "Hello world.\n"
        "This is a test.\n"
        "This is another test.\n"
        "This is a test v2."
    )
    document = Document(doc_text)
    llm = MagicMock(spec=BaseLLM)
    llm_predictor = MockLLMPredictor(max_tokens=256, llm=llm)
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

    # test tree index
    index = TreeIndex.from_documents([document], service_context=service_context)
    query_engine = index.as_query_engine()
    query_engine.query("What is?")

    # test keyword table index
    index_keyword = GPTKeywordTableIndex.from_documents(
        [document], service_context=service_context
    )
    query_engine = index_keyword.as_query_engine()
    query_engine.query("What is?")

    # test list index
    index_list = ListIndex.from_documents([document], service_context=service_context)
    query_engine = index_list.as_query_engine()
    query_engine.query("What is?")
