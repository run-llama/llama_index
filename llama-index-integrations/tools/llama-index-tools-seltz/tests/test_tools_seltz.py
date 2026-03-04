import os
from unittest.mock import Mock, patch

import pytest

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.seltz import SeltzToolSpec


def test_class():
    names_of_base_classes = [b.__name__ for b in SeltzToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_spec_functions():
    assert "search" in SeltzToolSpec.spec_functions


@patch("llama_index.tools.seltz.base.Seltz")
def test_init(mock_seltz):
    tool = SeltzToolSpec(api_key="test-key")
    mock_seltz.assert_called_once_with(api_key="test-key")
    assert tool.client == mock_seltz.return_value


@patch("llama_index.tools.seltz.base.Includes")
@patch("llama_index.tools.seltz.base.Seltz")
def test_search(mock_seltz, mock_includes_class):
    mock_doc1 = Mock()
    mock_doc1.content = "Result content 1"
    mock_doc1.url = "https://example1.com"

    mock_doc2 = Mock()
    mock_doc2.content = "Result content 2"
    mock_doc2.url = "https://example2.com"

    mock_response = Mock()
    mock_response.documents = [mock_doc1, mock_doc2]

    mock_client = Mock()
    mock_client.search.return_value = mock_response
    mock_seltz.return_value = mock_client

    mock_includes = Mock()
    mock_includes_class.return_value = mock_includes

    tool = SeltzToolSpec(api_key="test-key")
    results = tool.search("test query", max_documents=5)

    mock_includes_class.assert_called_once_with(max_documents=5)
    mock_client.search.assert_called_once_with("test query", includes=mock_includes)

    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)
    assert results[0].text == "Result content 1"
    assert results[0].metadata["url"] == "https://example1.com"
    assert results[1].text == "Result content 2"
    assert results[1].metadata["url"] == "https://example2.com"


@patch("llama_index.tools.seltz.base.Includes")
@patch("llama_index.tools.seltz.base.Seltz")
def test_search_default_max_documents(mock_seltz, mock_includes_class):
    mock_response = Mock()
    mock_response.documents = []

    mock_client = Mock()
    mock_client.search.return_value = mock_response
    mock_seltz.return_value = mock_client

    mock_includes = Mock()
    mock_includes_class.return_value = mock_includes

    tool = SeltzToolSpec(api_key="test-key")
    tool.search("test query")

    mock_includes_class.assert_called_once_with(max_documents=10)
    mock_client.search.assert_called_once_with("test query", includes=mock_includes)


@patch("llama_index.tools.seltz.base.Seltz")
def test_search_empty_results(mock_seltz):
    mock_response = Mock()
    mock_response.documents = []

    mock_client = Mock()
    mock_client.search.return_value = mock_response
    mock_seltz.return_value = mock_client

    tool = SeltzToolSpec(api_key="test-key")
    results = tool.search("no results query")

    assert results == []


# -- Integration tests --
# These tests hit the real Seltz API and require a valid SELTZ_API_KEY
# environment variable. They are skipped by default in CI.


@pytest.mark.skipif(
    not os.environ.get("SELTZ_API_KEY"),
    reason="SELTZ_API_KEY not set",
)
def test_integration_search():
    """Integration test: perform a real search against the Seltz API."""
    tool = SeltzToolSpec(api_key=os.environ["SELTZ_API_KEY"])
    results = tool.search("what is llama index?", max_documents=3)

    assert isinstance(results, list)
    assert len(results) > 0
    assert all(isinstance(doc, Document) for doc in results)
    assert all(doc.text for doc in results)
    assert all(doc.metadata.get("url") for doc in results)


@pytest.mark.skipif(
    not os.environ.get("SELTZ_API_KEY"),
    reason="SELTZ_API_KEY not set",
)
def test_integration_search_returns_documents():
    """Integration test: verify search returns well-formed Document objects."""
    tool = SeltzToolSpec(api_key=os.environ["SELTZ_API_KEY"])
    results = tool.search("artificial intelligence", max_documents=5)

    assert isinstance(results, list)
    assert len(results) > 0
    for doc in results:
        assert isinstance(doc, Document)
        assert doc.text
        assert doc.metadata.get("url")
