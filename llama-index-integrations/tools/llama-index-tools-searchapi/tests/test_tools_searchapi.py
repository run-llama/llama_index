"""Tests for SearchApiToolSpec.

All HTTP calls are mocked via unittest.mock — no real network access required.
"""

from unittest.mock import MagicMock, patch

import pytest

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.searchapi import SearchApiToolSpec

API_KEY = "test-api-key-12345"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ORGANIC_RESPONSE = {
    "organic_results": [
        {
            "title": "Result One",
            "link": "https://example.com/1",
            "snippet": "Snippet for result one.",
            "position": 1,
        },
        {
            "title": "Result Two",
            "link": "https://example.com/2",
            "snippet": "Snippet for result two.",
            "position": 2,
        },
    ]
}

NEWS_RESPONSE = {
    "organic_results": [
        {
            "title": "News Article",
            "link": "https://news.example.com/article",
            "snippet": "Breaking news snippet.",
            "source": "Example News",
            "date": "1 hour ago",
        }
    ]
}

SCHOLAR_RESPONSE = {
    "organic_results": [
        {
            "title": "A Great Paper",
            "link": "https://scholar.example.com/paper",
            "snippet": "Abstract of the paper.",
            "publication_info": {"summary": "Smith et al., 2024"},
            "inline_links": {"cited_by": {"total": 123}},
        }
    ]
}

IMAGE_RESPONSE = {
    "images_results": [
        {
            "title": "A Beautiful Image",
            "link": "https://page.example.com/image",
            "original": "https://cdn.example.com/image.jpg",
            "source": "example.com",
        }
    ]
}


# ---------------------------------------------------------------------------
# Class-level tests
# ---------------------------------------------------------------------------


def test_inherits_base_tool_spec():
    names = [b.__name__ for b in SearchApiToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names


def test_spec_functions():
    assert "search" in SearchApiToolSpec.spec_functions
    assert "news_search" in SearchApiToolSpec.spec_functions
    assert "scholar_search" in SearchApiToolSpec.spec_functions
    assert "image_search" in SearchApiToolSpec.spec_functions


# ---------------------------------------------------------------------------
# Initialisation tests
# ---------------------------------------------------------------------------


def test_init_with_explicit_key():
    tool = SearchApiToolSpec(api_key=API_KEY)
    assert tool.api_key == API_KEY
    assert tool.base_url == "https://www.searchapi.io/api/v1/search"


def test_init_reads_env_var(monkeypatch):
    monkeypatch.setenv("SEARCHAPI_API_KEY", "env-key-xyz")
    tool = SearchApiToolSpec()
    assert tool.api_key == "env-key-xyz"


def test_init_raises_without_key(monkeypatch):
    monkeypatch.delenv("SEARCHAPI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        SearchApiToolSpec()


def test_init_custom_base_url():
    tool = SearchApiToolSpec(api_key=API_KEY, base_url="https://custom.example.com")
    assert tool.base_url == "https://custom.example.com"


# ---------------------------------------------------------------------------
# search() tests
# ---------------------------------------------------------------------------


@patch("llama_index.tools.searchapi.base.httpx.get")
def test_search_returns_documents(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = ORGANIC_RESPONSE
    mock_get.return_value = mock_response

    tool = SearchApiToolSpec(api_key=API_KEY)
    docs = tool.search("python LLMs")

    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)
    assert docs[0].text == "Snippet for result one."
    assert docs[0].metadata["title"] == "Result One"
    assert docs[0].metadata["link"] == "https://example.com/1"
    assert docs[0].metadata["position"] == 1
    assert docs[1].metadata["position"] == 2


@patch("llama_index.tools.searchapi.base.httpx.get")
def test_search_passes_params(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {"organic_results": []}
    mock_get.return_value = mock_response

    tool = SearchApiToolSpec(api_key=API_KEY)
    tool.search("test query", num=5, gl="gb", hl="en", location="London")

    _, kwargs = mock_get.call_args
    params = kwargs["params"]
    assert params["q"] == "test query"
    assert params["engine"] == "google"
    assert params["num"] == 5
    assert params["gl"] == "gb"
    assert params["hl"] == "en"
    assert params["location"] == "London"


@patch("llama_index.tools.searchapi.base.httpx.get")
def test_search_omits_optional_params_when_none(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {"organic_results": []}
    mock_get.return_value = mock_response

    tool = SearchApiToolSpec(api_key=API_KEY)
    tool.search("test")

    _, kwargs = mock_get.call_args
    params = kwargs["params"]
    assert "gl" not in params
    assert "hl" not in params
    assert "location" not in params


@patch("llama_index.tools.searchapi.base.httpx.get")
def test_search_empty_results(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {}
    mock_get.return_value = mock_response

    tool = SearchApiToolSpec(api_key=API_KEY)
    docs = tool.search("something rare")
    assert docs == []


# ---------------------------------------------------------------------------
# news_search() tests
# ---------------------------------------------------------------------------


@patch("llama_index.tools.searchapi.base.httpx.get")
def test_news_search_returns_documents(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = NEWS_RESPONSE
    mock_get.return_value = mock_response

    tool = SearchApiToolSpec(api_key=API_KEY)
    docs = tool.news_search("AI regulation")

    assert len(docs) == 1
    assert docs[0].text == "Breaking news snippet."
    assert docs[0].metadata["source"] == "Example News"
    assert docs[0].metadata["date"] == "1 hour ago"
    assert docs[0].metadata["link"] == "https://news.example.com/article"


@patch("llama_index.tools.searchapi.base.httpx.get")
def test_news_search_uses_google_news_engine(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {"organic_results": []}
    mock_get.return_value = mock_response

    tool = SearchApiToolSpec(api_key=API_KEY)
    tool.news_search("latest tech")

    _, kwargs = mock_get.call_args
    assert kwargs["params"]["engine"] == "google_news"


# ---------------------------------------------------------------------------
# scholar_search() tests
# ---------------------------------------------------------------------------


@patch("llama_index.tools.searchapi.base.httpx.get")
def test_scholar_search_returns_documents(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = SCHOLAR_RESPONSE
    mock_get.return_value = mock_response

    tool = SearchApiToolSpec(api_key=API_KEY)
    docs = tool.scholar_search("retrieval augmented generation")

    assert len(docs) == 1
    assert docs[0].text == "Abstract of the paper."
    assert docs[0].metadata["title"] == "A Great Paper"
    assert docs[0].metadata["publication_info"] == "Smith et al., 2024"
    assert docs[0].metadata["cited_by_count"] == 123


@patch("llama_index.tools.searchapi.base.httpx.get")
def test_scholar_search_uses_google_scholar_engine(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {"organic_results": []}
    mock_get.return_value = mock_response

    tool = SearchApiToolSpec(api_key=API_KEY)
    tool.scholar_search("RAG survey")

    _, kwargs = mock_get.call_args
    assert kwargs["params"]["engine"] == "google_scholar"


@patch("llama_index.tools.searchapi.base.httpx.get")
def test_scholar_search_missing_inline_links(mock_get):
    """Gracefully handle results without inline_links or cited_by."""
    response = {
        "organic_results": [
            {
                "title": "Minimal Paper",
                "link": "https://scholar.example.com/p",
                "snippet": "Short abstract.",
                # no publication_info, no inline_links
            }
        ]
    }
    mock_response = MagicMock()
    mock_response.json.return_value = response
    mock_get.return_value = mock_response

    tool = SearchApiToolSpec(api_key=API_KEY)
    docs = tool.scholar_search("something")

    assert len(docs) == 1
    assert docs[0].metadata["publication_info"] == ""
    assert docs[0].metadata["cited_by_count"] == 0


# ---------------------------------------------------------------------------
# image_search() tests
# ---------------------------------------------------------------------------


@patch("llama_index.tools.searchapi.base.httpx.get")
def test_image_search_returns_documents(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = IMAGE_RESPONSE
    mock_get.return_value = mock_response

    tool = SearchApiToolSpec(api_key=API_KEY)
    docs = tool.image_search("golden gate bridge")

    assert len(docs) == 1
    assert docs[0].text == "A Beautiful Image"
    assert docs[0].metadata["original"] == "https://cdn.example.com/image.jpg"
    assert docs[0].metadata["source"] == "example.com"
    assert docs[0].metadata["link"] == "https://page.example.com/image"


@patch("llama_index.tools.searchapi.base.httpx.get")
def test_image_search_uses_google_images_engine(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {"images_results": []}
    mock_get.return_value = mock_response

    tool = SearchApiToolSpec(api_key=API_KEY)
    tool.image_search("cats")

    _, kwargs = mock_get.call_args
    assert kwargs["params"]["engine"] == "google_images"


# ---------------------------------------------------------------------------
# HTTP / auth tests
# ---------------------------------------------------------------------------


@patch("llama_index.tools.searchapi.base.httpx.get")
def test_request_sends_bearer_auth(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {"organic_results": []}
    mock_get.return_value = mock_response

    tool = SearchApiToolSpec(api_key=API_KEY)
    tool.search("test")

    _, kwargs = mock_get.call_args
    assert kwargs["headers"]["Authorization"] == f"Bearer {API_KEY}"


@patch("llama_index.tools.searchapi.base.httpx.get")
def test_request_raises_on_http_error(mock_get):
    import httpx

    mock_get.side_effect = httpx.HTTPStatusError(
        "401 Unauthorized",
        request=MagicMock(),
        response=MagicMock(status_code=401),
    )
    tool = SearchApiToolSpec(api_key=API_KEY)
    with pytest.raises(httpx.HTTPStatusError):
        tool.search("test")
