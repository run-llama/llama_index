"""Test token predictor."""

from typing import Any
from unittest.mock import patch

from llama_index.core.indices.keyword_table.base import KeywordTableIndex
from llama_index.core.indices.list.base import SummaryIndex
from llama_index.core.indices.tree.base import TreeIndex
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import Document

from tests.mock_utils.mock_text_splitter import mock_token_splitter_newline


@patch.object(TokenTextSplitter, "split_text", side_effect=mock_token_splitter_newline)
def test_token_predictor(mock_split: Any) -> None:
    """Test token predictor."""
    # here, just assert that token predictor runs (before checking behavior)
    # TODO: mock token counting a bit more carefully
    doc_text = (
        "Hello world.\nThis is a test.\nThis is another test.\nThis is a test v2."
    )
    document = Document(text=doc_text)

    # test tree index
    index = TreeIndex.from_documents([document])
    query_engine = index.as_query_engine()
    query_engine.query("What is?")

    # test keyword table index
    index_keyword = KeywordTableIndex.from_documents([document])
    query_engine = index_keyword.as_query_engine()
    query_engine.query("What is?")

    # test summary index
    index_list = SummaryIndex.from_documents([document])
    query_engine = index_list.as_query_engine()
    query_engine.query("What is?")
