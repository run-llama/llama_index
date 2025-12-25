"""Test token predictor."""

from typing import Any, List
from unittest.mock import patch

from llama_index.core.indices.keyword_table.base import KeywordTableIndex
from llama_index.core.indices.list.base import SummaryIndex
from llama_index.core.indices.tree.base import TreeIndex
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import Document
from llama_index.core.utils import get_tokenizer

from tests.mock_utils.mock_text_splitter import mock_token_splitter_newline


def mock_tokenizer_fn(text: str) -> List[str]:
    """
    Mock tokenizer function that splits text by spaces and newlines.
    
    This provides a predictable token counting mechanism for testing.
    Returns a list of tokens (words) from the input text.
    """
    if not text:
        return []
    # Split by spaces and newlines, filter out empty strings
    tokens = []
    for word in text.replace("\n", " ").split(" "):
        if word.strip():
            tokens.append(word.strip())
    return tokens


@patch.object(TokenTextSplitter, "split_text", side_effect=mock_token_splitter_newline)
@patch("llama_index.core.utils.get_tokenizer", return_value=mock_tokenizer_fn)
def test_token_predictor(mock_get_tokenizer: Any, mock_split: Any) -> None:
    """Test token predictor."""
    # Mock token counting by patching get_tokenizer to return our mock function.
    # This ensures consistent and predictable token counting behavior in tests,
    # replacing the TODO: mock token counting a bit more carefully
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
