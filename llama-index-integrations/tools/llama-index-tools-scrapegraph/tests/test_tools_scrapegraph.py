"""Unit tests for ScrapegraphAI tool specification."""

from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from llama_index.tools.scrapegraph import ScrapegraphToolSpec


class TestSchema(BaseModel):
    """Test schema for scraping operations."""

    title: str
    description: str


@pytest.fixture()
def tool_spec():
    """Create a ScrapegraphToolSpec instance for testing."""
    return ScrapegraphToolSpec()


@pytest.fixture()
def mock_sync_client():
    """Create a mock SyncClient."""
    with patch("llama_index.tools.scrapegraph.base.Client") as mock:
        client_instance = Mock()
        mock.return_value = client_instance
        yield client_instance


def test_sync_scraping(tool_spec: ScrapegraphToolSpec, mock_sync_client: Mock):
    """Test synchronous scraping functionality."""
    # Test data
    prompt = "Extract product information"
    url = "https://example.com"
    api_key = "sgai-0000-0000-0000-0000-0000-0000-0000-0000"
    schema = [TestSchema]
    expected_response = [{"title": "Test Product", "description": "Test Description"}]

    # Configure mock
    mock_sync_client.smartscraper.return_value = expected_response

    # Execute test
    response = tool_spec.scrapegraph_smartscraper(
        prompt=prompt, url=url, api_key=api_key, schema=schema
    )

    # Verify
    mock_sync_client.smartscraper.assert_called_once_with(
        website_url=url, user_prompt=prompt, output_schema=schema
    )
    assert response == expected_response
