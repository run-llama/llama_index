"""Tests for OutcrawlWebReader."""

from unittest.mock import Mock, patch

import pytest
import requests

from llama_index.core.schema import Document
from llama_index.readers.web import OutcrawlWebReader


def _response(payload: dict) -> Mock:
    response = Mock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def test_class_name() -> None:
    assert OutcrawlWebReader.class_name() == "OutcrawlWebReader"


def test_api_key_from_env(monkeypatch) -> None:
    monkeypatch.setenv("OUTCRAWL_API_KEY", "env-key")
    assert OutcrawlWebReader().api_key == "env-key"


def test_missing_api_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("OUTCRAWL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OUTCRAWL_API_KEY"):
        OutcrawlWebReader()


def test_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="Invalid mode"):
        OutcrawlWebReader(api_key="k", mode="crawl")


@patch("llama_index.readers.web.outcrawl_web.base.requests.post")
def test_scrape_returns_markdown_document(mock_post) -> None:
    mock_post.return_value = _response(
        {
            "url": "https://example.com/",
            "statusCode": 200,
            "markdown": "# Example Domain\n\nBody",
            "metadata": {"title": "Example Domain", "language": "en"},
            "usage": {"credits": "1"},
        }
    )
    reader = OutcrawlWebReader(api_key="test-key", params={"onlyMainContent": False})

    documents = reader.load_data(url="https://example.com", params={"maxAge": 0})

    mock_post.assert_called_once_with(
        "https://api.outcrawl.ai/v1/scrape",
        headers={
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json",
        },
        json={
            "url": "https://example.com",
            "formats": ["markdown"],
            "onlyMainContent": False,
            "maxAge": 0,
        },
        timeout=60,
    )
    assert documents == [
        Document(
            id_=documents[0].id_,
            text="# Example Domain\n\nBody",
            metadata={
                "source": "https://example.com/",
                "status_code": 200,
                "title": "Example Domain",
                "language": "en",
            },
        )
    ]


@patch("llama_index.readers.web.outcrawl_web.base.requests.post")
def test_search_returns_one_document_per_hit(mock_post) -> None:
    mock_post.return_value = _response(
        {
            "hits": [
                {"url": "https://a.dev", "title": "A", "snippet": "first", "rank": 1},
                {
                    "url": "https://b.dev",
                    "title": "B",
                    "snippet": "second",
                    "rank": 2,
                    "markdown": "# B page",
                },
            ],
            "usage": {"credits": "1"},
        }
    )
    reader = OutcrawlWebReader(api_key="test-key", mode="search")

    documents = reader.load_data(query="web readers", params={"scrape": True})

    assert mock_post.call_args.kwargs["json"] == {
        "query": "web readers",
        "scrape": True,
        "formats": ["markdown"],
    }
    assert [d.text for d in documents] == ["first", "# B page"]
    assert documents[1].metadata == {
        "source": "https://b.dev",
        "title": "B",
        "snippet": "second",
        "rank": 2,
        "query": "web readers",
    }


def test_missing_inputs_raise() -> None:
    with pytest.raises(ValueError, match="URL must be provided"):
        OutcrawlWebReader(api_key="k").load_data()
    with pytest.raises(ValueError, match="Query must be provided"):
        OutcrawlWebReader(api_key="k", mode="search").load_data()


@patch("llama_index.readers.web.outcrawl_web.base.requests.post")
def test_http_error_propagates(mock_post) -> None:
    response = Mock()
    response.raise_for_status.side_effect = requests.HTTPError("401 Unauthorized")
    mock_post.return_value = response

    with pytest.raises(requests.HTTPError):
        OutcrawlWebReader(api_key="bad").load_data(url="https://example.com")
