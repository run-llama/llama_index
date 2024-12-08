"""Unit tests for ScrapegraphAI tool specification."""

import pytest
from unittest.mock import Mock, patch
from pydantic import BaseModel
from llama_index.tools.scrapegraph.base import ScrapegraphToolSpec


class TestSchema(BaseModel):
    """Test schema for scraping operations."""
    title: str
    description: str


@pytest.fixture
def tool_spec():
    """Create a ScrapegraphToolSpec instance for testing."""
    return ScrapegraphToolSpec()


@pytest.fixture
def mock_client():
    """Create a mock Client."""
    with patch('llama_index.tools.scrapegraph.base.Client') as mock:
        client_instance = Mock()
        mock.return_value = client_instance
        yield client_instance


def test_smartscraper(tool_spec, mock_client):
    """Test smartscraper functionality."""
    # Test data
    prompt = "Extract product information"
    url = "https://example.com"
    api_key = "test_api_key"
    schema = [TestSchema]
    expected_response = [{"title": "Test Product", "description": "Test Description"}]

    # Configure mock
    mock_client.smartscraper.return_value = expected_response

    # Execute test
    response = tool_spec.scrapegraph_smartscraper(
        prompt=prompt,
        url=url,
        api_key=api_key,
        schema=schema
    )

    # Verify
    mock_client.smartscraper.assert_called_once_with(
        website_url=url,
        user_prompt=prompt
    )
    assert response == expected_response


def test_markdownify(tool_spec, mock_client):
    """Test markdownify functionality."""
    # Test data
    url = "https://example.com"
    api_key = "test_api_key"
    expected_response = "# Test Header\n\nTest content"

    # Configure mock
    mock_client.markdownify.return_value = expected_response

    # Execute test
    response = tool_spec.scrapegraph_markdownify(
        url=url,
        api_key=api_key
    )

    # Verify
    mock_client.markdownify.assert_called_once_with(
        website_url=url
    )
    assert response == expected_response


def test_local_scrape(tool_spec, mock_client):
    """Test local_scrape functionality."""
    # Test data
    text = "Sample text for scraping"
    api_key = "test_api_key"
    expected_response = {"extracted_data": "test data"}

    # Configure mock
    mock_client.local_scrape.return_value = expected_response

    # Execute test
    response = tool_spec.scrapegraph_local_scrape(
        text=text,
        api_key=api_key
    )

    # Verify
    mock_client.local_scrape.assert_called_once_with(
        text=text
    )
    assert response == expected_response


def test_missing_dependency():
    """Test handling of missing scrapegraph-py dependency."""
    with patch('importlib.util.find_spec', return_value=None):
        with pytest.raises(ImportError) as exc_info:
            ScrapegraphToolSpec()
        assert "requires the scrapegraph-py package" in str(exc_info.value)


def test_spec_functions_list(tool_spec):
    """Test that all required functions are in spec_functions."""
    expected_functions = [
        "scrapegraph_smartscraper",
        "scrapegraph_markdownify",
        "scrapegraph_local_scrape"
    ]
    assert all(func in tool_spec.spec_functions for func in expected_functions)


@pytest.mark.parametrize("method_name", [
    "scrapegraph_smartscraper",
    "scrapegraph_markdownify",
    "scrapegraph_local_scrape"
])
def test_method_existence(tool_spec, method_name):
    """Test that all specified methods exist in the tool spec."""
    assert hasattr(tool_spec, method_name)
    assert callable(getattr(tool_spec, method_name))


def test_client_initialization(mock_client):
    """Test that the Client is properly initialized with API key."""
    api_key = "test_api_key"
    tool_spec = ScrapegraphToolSpec()
    
    # Call any method that initializes the client
    tool_spec.scrapegraph_markdownify(url="https://example.com", api_key=api_key)
    
    # Verify client initialization
    mock_client.assert_called_once_with(api_key=api_key)
