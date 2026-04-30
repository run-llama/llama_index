"""Tests for PerplexitySearchToolSpec."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from llama_index.core.schema import Document
from llama_index.llms.perplexity import PerplexitySearchToolSpec


SAMPLE_RESPONSE = {
    "id": "abc",
    "results": [
        {
            "title": "Title One",
            "url": "https://example.com/1",
            "snippet": "Snippet one.",
            "date": "2025-02-10",
        },
        {
            "title": "Title Two",
            "url": "https://example.com/2",
            "snippet": "Snippet two.",
        },
    ],
}


def _mock_requests_response(status_code: int = 200, payload=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload if payload is not None else SAMPLE_RESPONSE
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(f"{status_code}")
    else:
        resp.raise_for_status.return_value = None
    return resp


def test_spec_functions_declared():
    spec = PerplexitySearchToolSpec(api_key="test")
    assert spec.spec_functions == ["perplexity_search"]
    assert hasattr(spec, "perplexity_search")


@patch("llama_index.llms.perplexity.search.requests.post")
def test_perplexity_search_returns_documents(mock_post):
    mock_post.return_value = _mock_requests_response()
    spec = PerplexitySearchToolSpec(api_key="test")
    docs = spec.perplexity_search("hello")

    assert isinstance(docs, list)
    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)
    assert docs[0].text == "Snippet one."
    assert docs[0].metadata["title"] == "Title One"
    assert docs[0].metadata["url"] == "https://example.com/1"
    assert docs[0].metadata["date"] == "2025-02-10"
    # second result has no date — should default to empty string, not crash
    assert docs[1].metadata["date"] == ""


@patch("llama_index.llms.perplexity.search.requests.post")
def test_filter_params_forwarded(mock_post):
    mock_post.return_value = _mock_requests_response()
    spec = PerplexitySearchToolSpec(api_key="test")
    spec.perplexity_search(
        "query",
        max_results=8,
        search_domain_filter=["arxiv.org"],
        search_recency_filter="week",
    )
    payload = mock_post.call_args.kwargs["json"]
    assert payload["max_results"] == 8
    assert payload["search_domain_filter"] == ["arxiv.org"]
    assert payload["search_recency_filter"] == "week"


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    monkeypatch.delenv("PPLX_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        PerplexitySearchToolSpec()


@patch("llama_index.llms.perplexity.search.requests.post")
def test_returns_to_tool_list(mock_post):
    """Tool spec should integrate with BaseToolSpec.to_tool_list()."""
    mock_post.return_value = _mock_requests_response()
    spec = PerplexitySearchToolSpec(api_key="test")
    tools = spec.to_tool_list()
    assert len(tools) == 1
    assert tools[0].metadata.name == "perplexity_search"
