"""Test list index."""

from typing import Any, List
from unittest.mock import patch

import pytest

from gpt_index.indices.list.base import GPTListIndex
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.schema import Document
from tests.mock_utils.mock_text_splitter import mock_token_splitter_newline


@pytest.fixture
def documents() -> List[Document]:
    """Get documents."""
    # NOTE: one document for now
    doc_text = (
        "Hello world.\n"
        "This is a test.\n"
        "This is another test.\n"
        "This is a test v2."
    )
    return [Document(doc_text)]


@patch.object(TokenTextSplitter, "split_text", side_effect=mock_token_splitter_newline)
def test_build_list(_mock_splitter: Any, documents: List[Document]) -> None:
    """Test build list."""
    list_index = GPTListIndex(documents)
    assert len(list_index.index_struct.nodes) == 4
    # check contents of nodes
    assert list_index.index_struct.nodes[0].text == "Hello world."
    assert list_index.index_struct.nodes[1].text == "This is a test."
    assert list_index.index_struct.nodes[2].text == "This is another test."
    assert list_index.index_struct.nodes[3].text == "This is a test v2."


@patch.object(TokenTextSplitter, "split_text", side_effect=mock_token_splitter_newline)
def test_list_insert(_mock_splitter: Any, documents: List[Document]) -> None:
    """Test insert to list."""
    list_index = GPTListIndex([])
    assert len(list_index.index_struct.nodes) == 0
    list_index.insert(documents[0])
    # check contents of nodes
    assert list_index.index_struct.nodes[0].text == "Hello world."
    assert list_index.index_struct.nodes[1].text == "This is a test."
    assert list_index.index_struct.nodes[2].text == "This is another test."
    assert list_index.index_struct.nodes[3].text == "This is a test v2."
