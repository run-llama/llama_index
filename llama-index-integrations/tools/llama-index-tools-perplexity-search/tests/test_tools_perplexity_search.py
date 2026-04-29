from typing import Any, Dict

import pytest
import requests
import requests_mock

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.perplexity_search import PerplexitySearchToolSpec
from llama_index.tools.perplexity_search.base import DEFAULT_ENDPOINT


SAMPLE_RESPONSE: Dict[str, Any] = {
    "id": "search-1",
    "results": [
        {
            "title": "Result one",
            "url": "https://example.com/1",
            "snippet": "Snippet one.",
            "date": "2026-04-01",
        },
        {
            "title": "Result two",
            "url": "https://example.com/2",
            "snippet": "Snippet two.",
            "date": "2026-04-02",
        },
    ],
}


def test_class_inheritance():
    names = [b.__name__ for b in PerplexitySearchToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names


def test_spec_functions():
    assert "perplexity_search" in PerplexitySearchToolSpec.spec_functions


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    monkeypatch.delenv("PPLX_API_KEY", raising=False)
    with pytest.raises(ValueError):
        PerplexitySearchToolSpec()


def test_falls_back_to_pplx_api_key_env(monkeypatch):
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    monkeypatch.setenv("PPLX_API_KEY", "fallback-key")
    tool = PerplexitySearchToolSpec()
    assert tool._api_key == "fallback-key"


def test_search_basic_returns_documents():
    tool = PerplexitySearchToolSpec(api_key="test-key")

    with requests_mock.Mocker() as m:
        m.post(DEFAULT_ENDPOINT, json=SAMPLE_RESPONSE)
        docs = tool.perplexity_search("hello world", max_results=2)

        assert m.call_count == 1
        sent = m.last_request.json()
        assert sent == {"query": "hello world", "max_results": 2}
        assert m.last_request.headers["Authorization"] == "Bearer test-key"

    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)
    assert docs[0].text == "Snippet one."
    assert docs[0].metadata["title"] == "Result one"
    assert docs[0].metadata["url"] == "https://example.com/1"
    assert docs[0].metadata["date"] == "2026-04-01"
    assert docs[1].metadata["url"] == "https://example.com/2"


def test_search_with_domain_filter():
    tool = PerplexitySearchToolSpec(api_key="test-key")

    with requests_mock.Mocker() as m:
        m.post(DEFAULT_ENDPOINT, json={"results": []})
        tool.perplexity_search(
            "site reliability",
            search_domain_filter=["nytimes.com", "-pinterest.com"],
        )
        sent = m.last_request.json()

    assert sent["search_domain_filter"] == ["nytimes.com", "-pinterest.com"]
    assert "search_recency_filter" not in sent


def test_search_with_recency_filter():
    tool = PerplexitySearchToolSpec(api_key="test-key")

    with requests_mock.Mocker() as m:
        m.post(DEFAULT_ENDPOINT, json={"results": []})
        tool.perplexity_search("breaking news", search_recency_filter="day")
        sent = m.last_request.json()

    assert sent["search_recency_filter"] == "day"
    assert "search_domain_filter" not in sent


def test_search_http_error_raises():
    tool = PerplexitySearchToolSpec(api_key="test-key")

    with requests_mock.Mocker() as m:
        m.post(DEFAULT_ENDPOINT, status_code=500, text="boom")
        with pytest.raises(requests.HTTPError):
            tool.perplexity_search("anything")


def test_search_handles_missing_optional_fields():
    tool = PerplexitySearchToolSpec(api_key="test-key")
    # No `date`, no `snippet` on second result.
    payload = {
        "results": [
            {"title": "T1", "url": "https://ex.com/1", "snippet": "S1"},
            {"title": "T2", "url": "https://ex.com/2"},
        ]
    }

    with requests_mock.Mocker() as m:
        m.post(DEFAULT_ENDPOINT, json=payload)
        docs = tool.perplexity_search("q")

    assert len(docs) == 2
    assert docs[0].metadata["date"] is None
    assert docs[1].text == ""
    assert docs[1].metadata["title"] == "T2"


def test_explicit_api_key_overrides_env(monkeypatch):
    monkeypatch.setenv("PERPLEXITY_API_KEY", "from-env")
    tool = PerplexitySearchToolSpec(api_key="explicit")
    assert tool._api_key == "explicit"
