"""Test keyword table index."""

from typing import Any, List, Optional, Set
from unittest.mock import patch

import pytest

from gpt_index.indices.keyword_table.simple_base import GPTSimpleKeywordTableIndex
from gpt_index.indices.keyword_table.utils import simple_extract_keywords
from gpt_index.langchain_helpers.text_splitter import TokenTextSplitter
from gpt_index.schema import Document
from tests.mock_utils.mock_text_splitter import mock_token_splitter_newline


def _mock_extract_keywords(
    text_chunk: str, max_keywords: Optional[int] = None, filter_stopwords: bool = True
) -> Set[str]:
    """Extract keywords (mock).

    Same as simple_extract_keywords but without filtering stopwords.

    """
    return simple_extract_keywords(
        text_chunk, max_keywords=max_keywords, filter_stopwords=False
    )


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
@patch(
    "gpt_index.indices.keyword_table.simple_base.simple_extract_keywords",
    _mock_extract_keywords,
)
def test_build_table(_mock_predict: Any, documents: List[Document]) -> None:
    """Test build table."""
    # test simple keyword table
    table = GPTSimpleKeywordTableIndex(documents)
    table_chunks = set(table.index_struct.text_chunks.values())
    assert len(table_chunks) == 4
    assert "Hello world." in table_chunks
    assert "This is a test." in table_chunks
    assert "This is another test." in table_chunks
    assert "This is a test v2." in table_chunks

    # test that expected keys are present in table
    # NOTE: in mock keyword extractor, stopwords are not filtered
    assert table.index_struct.table.keys() == {
        "this",
        "hello",
        "world",
        "test",
        "another",
        "v2",
        "is",
        "a",
        "v2",
    }


@patch.object(TokenTextSplitter, "split_text", side_effect=mock_token_splitter_newline)
@patch(
    "gpt_index.indices.keyword_table.simple_base.simple_extract_keywords",
    _mock_extract_keywords,
)
def test_insert(_mock_predict: Any, documents: List[Document]) -> None:
    """Test insert."""
    table = GPTSimpleKeywordTableIndex([])
    assert len(table.index_struct.table.keys()) == 0
    table.insert(documents[0])
    table_chunks = set(table.index_struct.text_chunks.values())
    assert "Hello world." in table_chunks
    assert "This is a test." in table_chunks
    assert "This is another test." in table_chunks
    assert "This is a test v2." in table_chunks
    # test that expected keys are present in table
    # NOTE: in mock keyword extractor, stopwords are not filtered
    assert table.index_struct.table.keys() == {
        "this",
        "hello",
        "world",
        "test",
        "another",
        "v2",
        "is",
        "a",
        "v2",
    }
