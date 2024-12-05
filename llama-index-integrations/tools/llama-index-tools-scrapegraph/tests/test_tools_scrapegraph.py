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
def mock_sync_client():
    """Create a mock SyncClient."""
    with patch('llama_index.tools.scrapegraph.base.SyncClient') as mock:
        client_instance = Mock()
        mock.return_value = client_instance
        yield client_instance


@pytest.fixture
def mock_async_client():
    """Create a mock AsyncClient."""
    with patch('llama_index.tools.scrapegraph.base.AsyncClient') as mock:
        client_instance = Mock()
        mock.return_value = client_instance
        yield client_instance


def test_feedback_submission(tool_spec, mock_sync_client):
    """Test feedback submission functionality."""
    # Test data
    request_id = "test_request_123"
    api_key = "test_api_key"
    rating = 5
    feedback_text = "Great results!"

    # Configure mock
    mock_sync_client.submit_feedback.return_value = "Feedback submitted successfully"

    # Execute test
    response = tool_spec.scrapegraph_feedback(
        request_id=request_id,
        api_key=api_key,
        rating=rating,
        feedback_text=feedback_text
    )

    # Verify
    mock_sync_client.submit_feedback.assert_called_once_with(
        request_id=request_id,
        rating=rating,
        feedback_text=feedback_text
    )
    mock_sync_client.close.assert_called_once()
    assert response == "Feedback submitted successfully"


def test_sync_scraping(tool_spec, mock_sync_client):
    """Test synchronous scraping functionality."""
    # Test data
    prompt = "Extract product information"
    url = "https://example.com"
    api_key = "test_api_key"
    schema = [TestSchema]
    expected_response = [{"title": "Test Product", "description": "Test Description"}]

    # Configure mock
    mock_sync_client.smartscraper.return_value = expected_response

    # Execute test
    response = tool_spec.scrapegraph_smartscraper_sync(
        prompt=prompt,
        url=url,
        api_key=api_key,
        schema=schema
    )

    # Verify
    mock_sync_client.smartscraper.assert_called_once_with(
        website_url=url,
        user_prompt=prompt,
        output_schema=schema
    )
    mock_sync_client.close.assert_called_once()
    assert response == expected_response


@pytest.mark.asyncio
async def test_async_scraping(tool_spec, mock_async_client):
    """Test asynchronous scraping functionality."""
    # Test data
    prompt = "Extract product information"
    url = "https://example.com"
    api_key = "test_api_key"
    schema = [TestSchema]
    expected_response = {"title": "Test Product", "description": "Test Description"}

    # Configure mock
    mock_async_client.smartscraper.return_value = expected_response

    # Execute test
    response = await tool_spec.scrapegraph_smartscraper_async(
        prompt=prompt,
        url=url,
        api_key=api_key,
        schema=schema
    )

    # Verify
    mock_async_client.smartscraper.assert_called_once_with(
        website_url=url,
        user_prompt=prompt,
        output_schema=schema
    )
    mock_async_client.close.assert_called_once()
    assert response == expected_response


def test_get_credits(tool_spec, mock_sync_client):
    """Test credit retrieval functionality."""
    # Test data
    api_key = "test_api_key"
    expected_credits = "500 credits remaining"

    # Configure mock
    mock_sync_client.get_credits.return_value = expected_credits

    # Execute test
    response = tool_spec.scrapegraph_get_credits(api_key=api_key)

    # Verify
    mock_sync_client.get_credits.assert_called_once()
    mock_sync_client.close.assert_called_once()
    assert response == expected_credits


def test_missing_dependency():
    """Test handling of missing scrapegraph-py dependency."""
    with patch('importlib.util.find_spec', return_value=None):
        with pytest.raises(ImportError) as exc_info:
            ScrapegraphToolSpec()
        assert "requires the scrapegraph-py package" in str(exc_info.value)
