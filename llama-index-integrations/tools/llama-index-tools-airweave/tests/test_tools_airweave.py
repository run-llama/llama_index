"""Tests for Airweave tool spec."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.airweave import AirweaveToolSpec


def test_class_inheritance() -> None:
    """Test that AirweaveToolSpec inherits from BaseToolSpec."""
    names_of_base_classes = [b.__name__ for b in AirweaveToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


@pytest.fixture()
def mock_airweave_sdk():
    """Create a mock Airweave SDK."""
    with patch("llama_index.tools.airweave.base.AirweaveSDK") as mock_sdk:
        yield mock_sdk


def test_initialization(mock_airweave_sdk) -> None:
    """Test AirweaveToolSpec initialization with required parameters."""
    tool_spec = AirweaveToolSpec(api_key="test-api-key")

    assert tool_spec is not None
    mock_airweave_sdk.assert_called_once_with(
        api_key="test-api-key",
        framework_name="llamaindex",
        framework_version="0.1.0",
    )


def test_initialization_with_base_url(mock_airweave_sdk) -> None:
    """Test initialization with custom base URL."""
    tool_spec = AirweaveToolSpec(
        api_key="test-api-key", base_url="https://custom.airweave.com"
    )

    assert tool_spec is not None
    mock_airweave_sdk.assert_called_once_with(
        api_key="test-api-key",
        framework_name="llamaindex",
        framework_version="0.1.0",
        base_url="https://custom.airweave.com",
    )


def test_spec_functions() -> None:
    """Test that all expected functions are in spec_functions."""
    expected_functions = [
        "search_collection",
        "advanced_search_collection",
        "search_and_generate_answer",
        "list_collections",
        "get_collection_info",
    ]
    assert AirweaveToolSpec.spec_functions == expected_functions


def test_search_collection(mock_airweave_sdk) -> None:
    """Test search_collection method with mock response."""
    # Setup mock client and response
    mock_client = MagicMock()
    mock_result = Mock()
    mock_result.content = "This is test content about LLMs"
    mock_result.score = 0.95
    mock_result.source = "test-source.pdf"
    mock_result.id = "result-123"
    mock_result.metadata = None  # Set to None to avoid Mock issues

    mock_response = Mock()
    mock_response.results = [mock_result]

    mock_client.collections.search.return_value = mock_response
    mock_airweave_sdk.return_value = mock_client

    # Create tool and test
    tool_spec = AirweaveToolSpec(api_key="test-key")
    results = tool_spec.search_collection(
        collection_id="test-collection", query="test query", limit=5
    )

    # Assertions
    assert len(results) == 1
    assert results[0].text == "This is test content about LLMs"
    assert results[0].metadata["score"] == 0.95
    assert results[0].metadata["source"] == "test-source.pdf"
    assert results[0].metadata["collection_id"] == "test-collection"
    assert results[0].metadata["result_id"] == "result-123"


def test_search_collection_empty_results(mock_airweave_sdk) -> None:
    """Test search_collection with no results."""
    mock_client = MagicMock()
    mock_response = Mock()
    mock_response.results = []

    mock_client.collections.search.return_value = mock_response
    mock_airweave_sdk.return_value = mock_client

    tool_spec = AirweaveToolSpec(api_key="test-key")
    results = tool_spec.search_collection(
        collection_id="test-collection", query="nonexistent query"
    )

    assert len(results) == 0


def test_advanced_search_collection(mock_airweave_sdk) -> None:
    """Test advanced_search_collection with all parameters."""
    mock_client = MagicMock()
    mock_result = Mock()
    mock_result.content = "Test content"
    mock_result.score = 0.95
    mock_result.metadata = None  # Set to None to avoid Mock issues

    mock_response = Mock()
    mock_response.results = [mock_result]
    mock_response.completion = "Generated answer from AI"

    mock_client.collections.search.return_value = mock_response
    mock_airweave_sdk.return_value = mock_client

    tool_spec = AirweaveToolSpec(api_key="test-key")
    result = tool_spec.advanced_search_collection(
        collection_id="test-collection",
        query="test query",
        limit=5,
        retrieval_strategy="hybrid",
        temporal_relevance=0.5,
        expand_query=True,
        rerank=True,
        generate_answer=True,
    )

    assert "documents" in result
    assert "answer" in result
    assert len(result["documents"]) == 1
    assert result["answer"] == "Generated answer from AI"


def test_advanced_search_no_answer(mock_airweave_sdk) -> None:
    """Test advanced_search_collection without answer generation."""
    mock_client = MagicMock()
    mock_result = Mock()
    mock_result.content = "Test content"
    mock_result.metadata = None  # Set to None to avoid Mock issues

    mock_response = Mock()
    mock_response.results = [mock_result]
    # Explicitly set completion to None
    mock_response.completion = None

    mock_client.collections.search.return_value = mock_response
    mock_airweave_sdk.return_value = mock_client

    tool_spec = AirweaveToolSpec(api_key="test-key")
    result = tool_spec.advanced_search_collection(
        collection_id="test-collection",
        query="test query",
        limit=5,
    )

    assert "documents" in result
    assert "answer" not in result
    assert len(result["documents"]) == 1


def test_search_and_generate_answer(mock_airweave_sdk) -> None:
    """Test search_and_generate_answer convenience method."""
    mock_client = MagicMock()
    mock_response = Mock()
    mock_response.completion = "This is the generated answer"

    mock_client.collections.search.return_value = mock_response
    mock_airweave_sdk.return_value = mock_client

    tool_spec = AirweaveToolSpec(api_key="test-key")
    answer = tool_spec.search_and_generate_answer(
        collection_id="test-collection",
        query="What is the answer?",
    )

    assert answer == "This is the generated answer"


def test_search_and_generate_answer_no_completion(mock_airweave_sdk) -> None:
    """Test search_and_generate_answer when no answer is generated."""
    mock_client = MagicMock()
    mock_response = Mock(spec=[])  # Create Mock with no attributes
    # Explicitly set completion to None
    mock_response.completion = None

    mock_client.collections.search.return_value = mock_response
    mock_airweave_sdk.return_value = mock_client

    tool_spec = AirweaveToolSpec(api_key="test-key")

    # Expect a UserWarning to be raised
    with pytest.warns(
        UserWarning, match="No answer could be generated from the search results"
    ):
        answer = tool_spec.search_and_generate_answer(
            collection_id="test-collection",
            query="What is the answer?",
        )

    assert answer is None


def test_list_collections(mock_airweave_sdk) -> None:
    """Test list_collections method."""
    # Setup mock
    mock_client = MagicMock()
    mock_collection1 = Mock()
    mock_collection1.readable_id = "finance-data"
    mock_collection1.name = "Finance Data"
    mock_collection1.created_at = "2024-01-01T00:00:00"

    mock_collection2 = Mock()
    mock_collection2.readable_id = "support-tickets"
    mock_collection2.name = "Support Tickets"
    mock_collection2.created_at = "2024-01-02T00:00:00"

    mock_client.collections.list.return_value = [mock_collection1, mock_collection2]
    mock_airweave_sdk.return_value = mock_client

    # Test
    tool_spec = AirweaveToolSpec(api_key="test-key")
    collections = tool_spec.list_collections()

    # Assertions
    assert len(collections) == 2
    assert collections[0]["id"] == "finance-data"
    assert collections[0]["name"] == "Finance Data"
    assert collections[1]["id"] == "support-tickets"
    assert collections[1]["name"] == "Support Tickets"


def test_get_collection_info(mock_airweave_sdk) -> None:
    """Test get_collection_info method."""
    # Setup mock
    mock_client = MagicMock()
    mock_collection = Mock()
    mock_collection.readable_id = "test-collection"
    mock_collection.name = "Test Collection"
    mock_collection.created_at = "2024-01-01T00:00:00"
    mock_collection.description = "A test collection"

    mock_client.collections.get.return_value = mock_collection
    mock_airweave_sdk.return_value = mock_client

    # Test
    tool_spec = AirweaveToolSpec(api_key="test-key")
    info = tool_spec.get_collection_info(collection_id="test-collection")

    # Assertions
    assert info["id"] == "test-collection"
    assert info["name"] == "Test Collection"
    assert info["description"] == "A test collection"
    mock_client.collections.get.assert_called_once_with(readable_id="test-collection")


def test_parse_dict_results(mock_airweave_sdk) -> None:
    """Test parsing search results when they come as dictionaries."""
    mock_client = MagicMock()
    mock_response = Mock()
    mock_response.results = [
        {
            "content": "Dict result content",
            "score": 0.88,
            "source": "dict-source",
            "id": "dict-123",
            "metadata": {"custom_field": "custom_value"},
        }
    ]

    mock_client.collections.search.return_value = mock_response
    mock_airweave_sdk.return_value = mock_client

    tool_spec = AirweaveToolSpec(api_key="test-key")
    results = tool_spec.search_collection(collection_id="test-collection", query="test")

    assert len(results) == 1
    assert results[0].text == "Dict result content"
    assert results[0].metadata["score"] == 0.88
    assert results[0].metadata["custom_field"] == "custom_value"
