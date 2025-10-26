"""Tests for SERPEX tool."""

import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from llama_index.tools.serpex import SerpexToolSpec


def test_serpex_init_with_key():
    """Test initialization with API key."""
    tool = SerpexToolSpec(api_key="test-key")
    assert tool.api_key == "test-key"
    assert tool.engine == "auto"


def test_serpex_init_with_custom_engine():
    """Test initialization with custom engine."""
    tool = SerpexToolSpec(api_key="test-key", engine="google")
    assert tool.api_key == "test-key"
    assert tool.engine == "google"


def test_serpex_init_without_key():
    """Test initialization without API key raises error."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="SERPEX_API_KEY not found"):
            SerpexToolSpec()


@patch("requests.get")
def test_search(mock_get):
    """Test search functionality."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {
                "title": "Test Result",
                "url": "https://example.com",
                "snippet": "Test snippet",
            }
        ],
        "metadata": {
            "number_of_results": 1,
            "response_time": 100,
        },
    }
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    tool = SerpexToolSpec(api_key="test-key")
    results = tool.search("test query")

    assert len(results) == 1
    assert "Test Result" in results[0].text
    assert results[0].text.startswith("Search results for")
    assert "URL: https://example.com" in results[0].text
    assert "Test snippet" in results[0].text


@patch("requests.get")
def test_search_with_engine(mock_get):
    """Test search with specific engine."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {
                "title": "DuckDuckGo Result",
                "url": "https://example.com",
                "snippet": "Privacy focused result",
            }
        ],
        "metadata": {
            "number_of_results": 1,
            "response_time": 150,
        },
    }
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    tool = SerpexToolSpec(api_key="test-key")
    results = tool.search("test query", engine="duckduckgo")

    assert len(results) == 1
    assert "DuckDuckGo Result" in results[0].text


@patch("requests.get")
def test_search_with_time_range(mock_get):
    """Test search with time range filter."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {
                "title": "Recent Result",
                "url": "https://example.com",
                "snippet": "Recent news",
            }
        ],
        "metadata": {
            "number_of_results": 1,
            "response_time": 120,
        },
    }
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    tool = SerpexToolSpec(api_key="test-key")
    results = tool.search("news", time_range="day", num_results=5)

    assert len(results) == 1
    assert "Recent Result" in results[0].text


@patch("requests.get")
def test_search_no_results(mock_get):
    """Test search with no results."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [],
        "metadata": {
            "number_of_results": 0,
            "response_time": 50,
        },
    }
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    tool = SerpexToolSpec(api_key="test-key")
    results = tool.search("nonexistent query")

    assert len(results) == 1
    assert "No search results found" in results[0].text


@patch("requests.get")
def test_search_api_error(mock_get):
    """Test search handles API errors."""
    mock_get.side_effect = requests.exceptions.RequestException("API Error")

    tool = SerpexToolSpec(api_key="test-key")
    results = tool.search("test query")

    assert len(results) == 1
    assert "Error calling SERPEX API" in results[0].text


# Integration test (requires real API key)
@pytest.mark.skip(reason="Requires real SERPEX API key")
def test_real_search():
    """Test real search with actual API."""
    api_key = os.environ.get("SERPEX_API_KEY")
    if not api_key:
        pytest.skip("SERPEX_API_KEY not set")

    tool = SerpexToolSpec(api_key=api_key)
    results = tool.search("LlamaIndex", num_results=5)

    assert len(results) > 0
    assert "LlamaIndex" in results[0].text or "llama" in results[0].text.lower()


@pytest.mark.skip(reason="Requires real SERPEX API key")
def test_real_search_with_filters():
    """Test real search with filters."""
    api_key = os.environ.get("SERPEX_API_KEY")
    if not api_key:
        pytest.skip("SERPEX_API_KEY not set")

    tool = SerpexToolSpec(api_key=api_key, engine="duckduckgo")
    results = tool.search("AI news", num_results=3, time_range="week")

    assert len(results) > 0
