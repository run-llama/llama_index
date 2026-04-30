"""Tests for the PerplexitySearch low-level client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import requests

from llama_index.llms.perplexity.search import PerplexitySearch, _get_package_version


SAMPLE_RESPONSE = {
    "id": "abc123",
    "results": [
        {
            "title": "Example Page",
            "url": "https://example.com/a",
            "snippet": "An example snippet.",
            "date": "2025-01-15",
        },
        {
            "title": "Another",
            "url": "https://example.com/b",
            "snippet": "Another snippet.",
        },
    ],
}


def _mock_requests_response(status_code: int = 200, payload=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload if payload is not None else SAMPLE_RESPONSE
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(f"{status_code} error")
    else:
        resp.raise_for_status.return_value = None
    return resp


def _mock_httpx_response(status_code: int = 200, payload=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload if payload is not None else SAMPLE_RESPONSE
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"{status_code} error", request=MagicMock(), response=MagicMock()
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


@patch("llama_index.llms.perplexity.search.requests.post")
def test_search_happy_path(mock_post):
    mock_post.return_value = _mock_requests_response()
    client = PerplexitySearch(api_key="test-key")
    results = client.search("hello world")

    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0]["title"] == "Example Page"
    assert results[0]["url"] == "https://example.com/a"
    assert results[0]["snippet"] == "An example snippet."

    args, kwargs = mock_post.call_args
    assert args[0] == "https://api.perplexity.ai/search"
    assert kwargs["json"]["query"] == "hello world"
    assert kwargs["json"]["max_results"] == 5


def test_env_var_fallback_perplexity(monkeypatch):
    monkeypatch.delenv("PPLX_API_KEY", raising=False)
    monkeypatch.setenv("PERPLEXITY_API_KEY", "from-perplexity-env")
    client = PerplexitySearch()
    assert client.api_key == "from-perplexity-env"


def test_env_var_fallback_pplx(monkeypatch):
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    monkeypatch.setenv("PPLX_API_KEY", "from-pplx-env")
    client = PerplexitySearch()
    assert client.api_key == "from-pplx-env"


def test_env_var_perplexity_takes_precedence(monkeypatch):
    monkeypatch.setenv("PERPLEXITY_API_KEY", "primary")
    monkeypatch.setenv("PPLX_API_KEY", "fallback")
    client = PerplexitySearch()
    assert client.api_key == "primary"


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    monkeypatch.delenv("PPLX_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        PerplexitySearch()


@patch("llama_index.llms.perplexity.search.requests.post")
def test_filter_params_forwarded(mock_post):
    mock_post.return_value = _mock_requests_response()
    client = PerplexitySearch(api_key="test-key")
    client.search(
        "query",
        max_results=10,
        search_domain_filter=["nytimes.com", "-pinterest.com"],
        search_recency_filter="month",
    )

    payload = mock_post.call_args.kwargs["json"]
    assert payload["query"] == "query"
    assert payload["max_results"] == 10
    assert payload["search_domain_filter"] == ["nytimes.com", "-pinterest.com"]
    assert payload["search_recency_filter"] == "month"


@patch("llama_index.llms.perplexity.search.requests.post")
def test_attribution_header_present(mock_post):
    mock_post.return_value = _mock_requests_response()
    client = PerplexitySearch(api_key="test-key")
    client.search("q")

    headers = mock_post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["Content-Type"] == "application/json"
    assert headers["X-Pplx-Integration"] == f"llamaindex/{_get_package_version()}"
    assert headers["X-Pplx-Integration"].startswith("llamaindex/")


@patch("llama_index.llms.perplexity.search.requests.post")
def test_4xx_error_raises(mock_post):
    mock_post.return_value = _mock_requests_response(status_code=401)
    client = PerplexitySearch(api_key="bad-key")
    with pytest.raises(requests.HTTPError):
        client.search("q")


@patch("llama_index.llms.perplexity.search.requests.post")
def test_5xx_error_raises(mock_post):
    mock_post.return_value = _mock_requests_response(status_code=500)
    client = PerplexitySearch(api_key="test-key")
    with pytest.raises(requests.HTTPError):
        client.search("q")


@pytest.mark.asyncio
async def test_async_search_happy_path():
    mock_response = _mock_httpx_response()
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "llama_index.llms.perplexity.search.httpx.AsyncClient",
        return_value=mock_client,
    ):
        client = PerplexitySearch(api_key="test-key")
        results = await client.asearch(
            "async query",
            max_results=3,
            search_recency_filter="day",
        )

    assert len(results) == 2
    assert results[0]["title"] == "Example Page"

    call_kwargs = mock_client.post.call_args.kwargs
    assert call_kwargs["json"]["query"] == "async query"
    assert call_kwargs["json"]["max_results"] == 3
    assert call_kwargs["json"]["search_recency_filter"] == "day"
    assert (
        call_kwargs["headers"]["X-Pplx-Integration"]
        == f"llamaindex/{_get_package_version()}"
    )


@pytest.mark.asyncio
async def test_async_search_5xx_raises():
    mock_response = _mock_httpx_response(status_code=503)
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "llama_index.llms.perplexity.search.httpx.AsyncClient",
        return_value=mock_client,
    ):
        client = PerplexitySearch(api_key="test-key")
        with pytest.raises(httpx.HTTPStatusError):
            await client.asearch("q")


@patch("llama_index.llms.perplexity.search.requests.post")
def test_extracts_results_from_list_response(mock_post):
    """Some servers may return a bare list — handle gracefully."""
    mock_post.return_value = _mock_requests_response(
        payload=[{"title": "t", "url": "u", "snippet": "s"}]
    )
    client = PerplexitySearch(api_key="test-key")
    results = client.search("q")
    assert results == [{"title": "t", "url": "u", "snippet": "s"}]
