from unittest.mock import Mock, patch
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.core.schema import Document
from llama_index.tools.tavily_research import TavilyToolSpec


def test_class():
    names_of_base_classes = [b.__name__ for b in TavilyToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_spec_functions():
    """Test that spec_functions includes both search and extract methods."""
    assert "search" in TavilyToolSpec.spec_functions
    assert "extract" in TavilyToolSpec.spec_functions


@patch("tavily.TavilyClient")
def test_init(mock_tavily_client):
    """Test TavilyToolSpec initialization."""
    api_key = "test_api_key"
    tool = TavilyToolSpec(api_key=api_key)

    mock_tavily_client.assert_called_once_with(api_key=api_key)
    assert tool.client == mock_tavily_client.return_value


@patch("tavily.TavilyClient")
def test_search(mock_tavily_client):
    """Test search method returns properly formatted Document objects."""
    # Setup mock response
    mock_response = {
        "results": [
            {"content": "Test content 1", "url": "https://example1.com"},
            {"content": "Test content 2", "url": "https://example2.com"},
        ]
    }

    mock_client_instance = Mock()
    mock_client_instance.search.return_value = mock_response
    mock_tavily_client.return_value = mock_client_instance

    # Create tool and call search
    tool = TavilyToolSpec(api_key="test_key")
    results = tool.search("test query", max_results=5)

    # Verify client.search was called correctly
    mock_client_instance.search.assert_called_once_with(
        "test query", max_results=5, search_depth="advanced"
    )

    # Verify results
    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)

    assert results[0].text == "Test content 1"
    assert results[0].extra_info["url"] == "https://example1.com"

    assert results[1].text == "Test content 2"
    assert results[1].extra_info["url"] == "https://example2.com"


@patch("tavily.TavilyClient")
def test_search_with_default_max_results(mock_tavily_client):
    """Test search method uses default max_results of 6."""
    mock_response = {"results": []}

    mock_client_instance = Mock()
    mock_client_instance.search.return_value = mock_response
    mock_tavily_client.return_value = mock_client_instance

    tool = TavilyToolSpec(api_key="test_key")
    tool.search("test query")

    mock_client_instance.search.assert_called_once_with(
        "test query", max_results=6, search_depth="advanced"
    )


@patch("tavily.TavilyClient")
def test_extract(mock_tavily_client):
    """Test extract method returns properly formatted Document objects."""
    # Setup mock response
    mock_response = {
        "results": [
            {
                "raw_content": "Extracted content 1",
                "url": "https://example1.com",
                "favicon": "https://example1.com/favicon.ico",
                "images": ["https://example1.com/image1.jpg"],
            },
            {
                "raw_content": "Extracted content 2",
                "url": "https://example2.com",
                "favicon": "https://example2.com/favicon.ico",
                "images": ["https://example2.com/image2.jpg"],
            },
        ]
    }

    mock_client_instance = Mock()
    mock_client_instance.extract.return_value = mock_response
    mock_tavily_client.return_value = mock_client_instance

    # Create tool and call extract
    tool = TavilyToolSpec(api_key="test_key")
    urls = ["https://example1.com", "https://example2.com"]
    results = tool.extract(
        urls=urls,
        include_images=True,
        include_favicon=True,
        extract_depth="advanced",
        format="text",
    )

    # Verify client.extract was called correctly
    mock_client_instance.extract.assert_called_once_with(
        urls,
        include_images=True,
        include_favicon=True,
        extract_depth="advanced",
        format="text",
    )

    # Verify results
    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)

    assert results[0].text == "Extracted content 1"
    assert results[0].extra_info["url"] == "https://example1.com"
    assert results[0].extra_info["favicon"] == "https://example1.com/favicon.ico"
    assert results[0].extra_info["images"] == ["https://example1.com/image1.jpg"]

    assert results[1].text == "Extracted content 2"
    assert results[1].extra_info["url"] == "https://example2.com"


@patch("tavily.TavilyClient")
def test_extract_with_defaults(mock_tavily_client):
    """Test extract method uses correct default parameters."""
    mock_response = {"results": []}

    mock_client_instance = Mock()
    mock_client_instance.extract.return_value = mock_response
    mock_tavily_client.return_value = mock_client_instance

    tool = TavilyToolSpec(api_key="test_key")
    urls = ["https://example.com"]
    tool.extract(urls)

    mock_client_instance.extract.assert_called_once_with(
        urls,
        include_images=False,
        include_favicon=False,
        extract_depth="basic",
        format="markdown",
    )


@patch("tavily.TavilyClient")
def test_extract_empty_results(mock_tavily_client):
    """Test extract method handles empty results gracefully."""
    mock_response = {"results": []}

    mock_client_instance = Mock()
    mock_client_instance.extract.return_value = mock_response
    mock_tavily_client.return_value = mock_client_instance

    tool = TavilyToolSpec(api_key="test_key")
    results = tool.extract(urls=["https://example.com"])

    assert results == []


@patch("tavily.TavilyClient")
def test_extract_missing_fields(mock_tavily_client):
    """Test extract method handles missing fields in response."""
    # Mock response with missing fields
    mock_response = {
        "results": [
            {
                "url": "https://example.com"
                # Missing raw_content, favicon, images
            }
        ]
    }

    mock_client_instance = Mock()
    mock_client_instance.extract.return_value = mock_response
    mock_tavily_client.return_value = mock_client_instance

    tool = TavilyToolSpec(api_key="test_key")
    results = tool.extract(urls=["https://example.com"])

    assert len(results) == 1
    assert results[0].text == ""  # Empty string for missing raw_content
    assert results[0].extra_info["url"] == "https://example.com"
    assert results[0].extra_info["favicon"] is None
    assert results[0].extra_info["images"] is None


@patch("tavily.TavilyClient")
def test_extract_no_results_key(mock_tavily_client):
    """Test extract method handles response without 'results' key."""
    mock_response = {}  # No 'results' key

    mock_client_instance = Mock()
    mock_client_instance.extract.return_value = mock_response
    mock_tavily_client.return_value = mock_client_instance

    tool = TavilyToolSpec(api_key="test_key")
    results = tool.extract(urls=["https://example.com"])

    assert results == []
