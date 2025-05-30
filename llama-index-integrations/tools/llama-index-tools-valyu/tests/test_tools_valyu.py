from unittest.mock import Mock, patch
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.valyu import ValyuToolSpec
from llama_index.core.schema import Document


def test_class():
    names_of_base_classes = [b.__name__ for b in ValyuToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


@patch("valyu.Valyu")
def test_init(mock_valyu):
    """Test ValyuToolSpec initialization."""
    mock_client = Mock()
    mock_valyu.return_value = mock_client

    tool = ValyuToolSpec(api_key="test_key", verbose=True, max_price=50.0)

    assert tool.client == mock_client
    assert tool._verbose is True
    assert tool._max_price == 50.0
    mock_valyu.assert_called_once_with(api_key="test_key")


@patch("valyu.Valyu")
def test_search_with_default_max_price(mock_valyu):
    """Test search method when max_price is None (uses default)."""
    # Mock the Valyu client and response
    mock_client = Mock()
    mock_valyu.return_value = mock_client

    # Create mock result object
    mock_result = Mock()
    mock_result.content = "Test content"
    mock_result.title = "Test title"
    mock_result.url = "https://test.com"
    mock_result.source = "test_source"
    mock_result.price = 1.0
    mock_result.length = 100
    mock_result.data_type = "text"
    mock_result.relevance_score = 0.8

    # Mock response object
    mock_response = Mock()
    mock_response.results = [mock_result]
    mock_client.search.return_value = mock_response

    tool = ValyuToolSpec(api_key="test_key", max_price=75.0)

    # Test search with max_price=None (should use default)
    documents = tool.search(query="test query", max_price=None)

    # Verify the client was called with the default max_price
    mock_client.search.assert_called_once_with(
        query="test query",
        search_type="all",
        max_num_results=5,
        relevance_threshold=0.5,
        max_price=75.0,  # Should use the default from init
        start_date=None,
        end_date=None,
    )

    # Verify document creation
    assert len(documents) == 1
    assert isinstance(documents[0], Document)
    assert documents[0].text == "Test content"
    assert documents[0].metadata["title"] == "Test title"
    assert documents[0].metadata["url"] == "https://test.com"
    assert documents[0].metadata["source"] == "test_source"
    assert documents[0].metadata["price"] == 1.0
    assert documents[0].metadata["length"] == 100
    assert documents[0].metadata["data_type"] == "text"
    assert documents[0].metadata["relevance_score"] == 0.8


@patch("valyu.Valyu")
@patch("builtins.print")
def test_search_with_verbose(mock_print, mock_valyu):
    """Test search method with verbose=True."""
    # Mock the Valyu client and response
    mock_client = Mock()
    mock_valyu.return_value = mock_client

    # Create mock result object
    mock_result = Mock()
    mock_result.content = "Verbose test content"
    mock_result.title = "Verbose test title"
    mock_result.url = "https://verbose-test.com"
    mock_result.source = "verbose_source"
    mock_result.price = 2.0
    mock_result.length = 200
    mock_result.data_type = "text"
    mock_result.relevance_score = 0.9

    # Mock response object
    mock_response = Mock()
    mock_response.results = [mock_result]
    mock_client.search.return_value = mock_response

    tool = ValyuToolSpec(api_key="test_key", verbose=True)

    # Test search with verbose output
    documents = tool.search(query="verbose test query", max_price=25.0)

    # Verify verbose print was called
    mock_print.assert_called_once_with(f"[Valyu Tool] Response: {mock_response}")

    # Verify document creation
    assert len(documents) == 1
    assert documents[0].text == "Verbose test content"


@patch("valyu.Valyu")
def test_search_with_custom_parameters(mock_valyu):
    """Test search method with custom parameters."""
    # Mock the Valyu client and response
    mock_client = Mock()
    mock_valyu.return_value = mock_client

    # Create mock result object
    mock_result = Mock()
    mock_result.content = "Custom test content"
    mock_result.title = "Custom test title"
    mock_result.url = "https://custom-test.com"
    mock_result.source = "custom_source"
    mock_result.price = 3.0
    mock_result.length = 300
    mock_result.data_type = "text"
    mock_result.relevance_score = 0.7

    # Mock response object
    mock_response = Mock()
    mock_response.results = [mock_result]
    mock_client.search.return_value = mock_response

    tool = ValyuToolSpec(api_key="test_key")

    # Test search with custom parameters
    documents = tool.search(
        query="custom test query",
        search_type="proprietary",
        max_num_results=10,
        relevance_threshold=0.8,
        max_price=50.0,
        start_date="2023-01-01",
        end_date="2023-12-31",
    )

    # Verify the client was called with custom parameters
    mock_client.search.assert_called_once_with(
        query="custom test query",
        search_type="proprietary",
        max_num_results=10,
        relevance_threshold=0.8,
        max_price=50.0,
        start_date="2023-01-01",
        end_date="2023-12-31",
    )

    # Verify document creation
    assert len(documents) == 1
    assert documents[0].text == "Custom test content"


@patch("valyu.Valyu")
def test_search_multiple_results(mock_valyu):
    """Test search method with multiple results."""
    # Mock the Valyu client and response
    mock_client = Mock()
    mock_valyu.return_value = mock_client

    # Create multiple mock result objects
    mock_result1 = Mock()
    mock_result1.content = "First result content"
    mock_result1.title = "First title"
    mock_result1.url = "https://first.com"
    mock_result1.source = "first_source"
    mock_result1.price = 1.0
    mock_result1.length = 100
    mock_result1.data_type = "text"
    mock_result1.relevance_score = 0.9

    mock_result2 = Mock()
    mock_result2.content = "Second result content"
    mock_result2.title = "Second title"
    mock_result2.url = "https://second.com"
    mock_result2.source = "second_source"
    mock_result2.price = 2.0
    mock_result2.length = 200
    mock_result2.data_type = "text"
    mock_result2.relevance_score = 0.8

    # Mock response object
    mock_response = Mock()
    mock_response.results = [mock_result1, mock_result2]
    mock_client.search.return_value = mock_response

    tool = ValyuToolSpec(api_key="test_key")

    # Test search with multiple results
    documents = tool.search(query="multi result query")

    # Verify multiple documents were created
    assert len(documents) == 2
    assert documents[0].text == "First result content"
    assert documents[1].text == "Second result content"
    assert documents[0].metadata["title"] == "First title"
    assert documents[1].metadata["title"] == "Second title"
