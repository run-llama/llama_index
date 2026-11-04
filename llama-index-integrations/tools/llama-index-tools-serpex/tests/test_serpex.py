"""
Tests for the Serpex tool.

tests/fixtures holds sanitized responses captured from api.serpex.dev on
2026-10-23 (statuses that cannot be triggered safely use the exact body from
the backend code); manifest.json has each status and headers. No network:
``requests.get`` is patched.
"""

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
import requests

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.serpex import SerpexToolSpec
from llama_index.tools.serpex.base import USER_AGENT, SerpexAPIError

FIXTURES = Path(__file__).parent / "fixtures"
MANIFEST: Dict[str, Any] = json.loads((FIXTURES / "manifest.json").read_text())
NEVER_SENT = ("engine", "engines", "category", "time_range", "num", "num_results")


def fixture(name: str) -> Any:
    return json.loads((FIXTURES / f"{name}.json").read_text())


class FakeResponse:
    def __init__(self, name: str) -> None:
        meta = MANIFEST[name]
        self.status_code = meta["status"]
        self.ok = 200 <= self.status_code < 300
        self.reason = ""
        self.headers = meta["headers"]
        self._body = fixture(name)

    def json(self) -> Any:
        return self._body


def serve(name: str) -> Any:
    """Patch requests.get to answer with one fixture; returns the mock."""
    return patch("llama_index.tools.serpex.base.requests.get", return_value=FakeResponse(name))


def test_class():
    names_of_base_classes = [b.__name__ for b in SerpexToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_serpex_init_with_key():
    tool = SerpexToolSpec(api_key="test-key")
    assert tool.api_key == "test-key"
    assert tool.engine is None
    assert tool.include_content is False
    assert tool.content_results == 5


def test_serpex_init_deprecated_engine_is_kept_and_warned():
    with pytest.warns(DeprecationWarning):
        tool = SerpexToolSpec(api_key="test-key", engine="legacy-value")
    assert tool.engine == "legacy-value"


def test_serpex_init_without_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="SERPEX_API_KEY not found"):
            SerpexToolSpec()


def test_content_results_must_be_5_or_10():
    with pytest.raises(ValueError, match="5 or 10"):
        SerpexToolSpec(api_key="test-key", content_results=7)


def test_tool_schema_exposes_only_real_arguments():
    """The agent sees query and num_results; deprecated arguments stay hidden."""
    tools = SerpexToolSpec(api_key="test-key").to_tool_list()
    assert [t.metadata.name for t in tools] == ["search"]
    schema = tools[0].metadata.get_parameters_dict()
    assert set(schema["properties"]) == {"query", "num_results"}
    assert schema["required"] == ["query"]
    text = json.dumps(schema).lower() + tools[0].metadata.description.lower()
    for word in ("multi-engine", "serp api", "cheapest", " engine"):
        assert word not in text


def test_search_sends_only_q():
    with serve("search_plain") as get:
        docs = SerpexToolSpec(api_key="test-key").search("python asyncio tutorial")
    args, kwargs = get.call_args
    assert args[0] == "https://api.serpex.dev/api/search"
    assert kwargs["params"] == {"q": "python asyncio tutorial"}
    assert kwargs["headers"] == {
        "Authorization": "Bearer test-key",
        "User-Agent": USER_AGENT,
    }
    assert kwargs["timeout"] >= 60
    assert len(docs) == 10

    first = fixture("search_plain")["results"][0]
    assert docs[0].text == f"{first['title']}\nURL: {first['url']}\n{first['snippet']}"
    assert docs[0].metadata == {
        "title": first["title"],
        "url": first["url"],
        "snippet": first["snippet"],
        "position": 1,
        "number_of_results": 10,
        "response_time": fixture("search_plain")["metadata"]["response_time"],
        "credits_used": 1,
        "query": "python asyncio tutorial",
        "engine": "auto",
    }


def test_deprecated_search_arguments_are_accepted_warned_and_not_sent():
    with serve("search_repeat_deprecated_engine") as get:
        with pytest.warns(DeprecationWarning):
            docs = SerpexToolSpec(api_key="test-key").search(
                "python asyncio tutorial",
                num_results=3,
                engine="legacy-a",
                time_range="day",
            )
    params = get.call_args[1]["params"]
    assert params == {"q": "python asyncio tutorial"}
    for name in NEVER_SENT:
        assert name not in params
    assert len(docs) == 3


def test_no_warning_without_deprecated_arguments():
    with serve("search_plain"):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            SerpexToolSpec(api_key="test-key").search("x")


@pytest.mark.parametrize("num_results,expected", [(1, 1), (5, 5), (10, 10), (100, 10), (0, 10)])
def test_num_results_caps_client_side(num_results: int, expected: int):
    with serve("search_plain"):
        docs = SerpexToolSpec(api_key="test-key").search("x", num_results=num_results)
    assert len(docs) == expected


def test_include_content_puts_page_markdown_in_the_document():
    with serve("search_content") as get:
        docs = SerpexToolSpec(api_key="test-key", include_content=True).search("rag")
    kwargs = get.call_args[1]
    assert kwargs["params"] == {
        "q": "rag",
        "include_content": "true",
        "content_results": 5,
    }
    assert kwargs["timeout"] >= 100
    raw = fixture("search_content")["results"]
    for i, doc in enumerate(docs):
        if i < 5:
            assert doc.text.endswith(raw[i]["content"])
        else:
            assert "\n\n" not in doc.text
        assert "content_error" not in doc.metadata


def test_content_errors_are_kept_in_metadata():
    with serve("search_content_errors"):
        docs = SerpexToolSpec(api_key="test-key", include_content=True, content_results=10).search(
            "x"
        )
    errors: List[str] = [d.metadata.get("content_error") for d in docs[:5]]
    assert errors == ["robots_disallowed"] * 5
    assert all("\n\n" not in d.text for d in docs)


def test_search_no_results():
    with serve("search_no_results"):
        assert SerpexToolSpec(api_key="test-key").search("x") == []


@pytest.mark.parametrize(
    "name,status,text,retryable",
    [
        (
            "error_400_content_results",
            400,
            "Serpex rejected the request (HTTP 400): content_results must be 5 or 10.",
            False,
        ),
        (
            "error_401_bad_key",
            401,
            "Serpex rejected the API key (HTTP 401): Invalid or inactive API key. "
            "Check SERPEX_API_KEY.",
            False,
        ),
        (
            "error_402_search",
            402,
            "Out of Serpex credits (HTTP 402). Balance: 0 credits.",
            False,
        ),
        (
            "error_429",
            429,
            "Serpex rate limit exceeded (limit: 5 requests per second). Retry in 292 ms.",
            True,
        ),
        (
            "error_503_search",
            503,
            "Serpex is temporarily unavailable (HTTP 503): "
            "Search temporarily unavailable, please retry. Not charged.",
            True,
        ),
        (
            "error_500_search",
            500,
            "Serpex is temporarily unavailable (HTTP 500): "
            "Internal server error. Not charged. Retry shortly.",
            True,
        ),
        ("error_500_malformed_json", 500, "Unexpected end of JSON input.", True),
    ],
)
def test_api_errors_are_clear_and_still_http_errors(
    name: str, status: int, text: str, retryable: bool
):
    with serve(name):
        with pytest.raises(requests.exceptions.HTTPError) as info:
            SerpexToolSpec(api_key="test-key").search("x")
    err = info.value
    assert isinstance(err, SerpexAPIError)
    assert err.status_code == status and err.retryable is retryable
    assert text in str(err)
    assert err.details == fixture(name)


def test_network_errors_propagate():
    with patch(
        "llama_index.tools.serpex.base.requests.get",
        side_effect=requests.exceptions.ConnectionError("connection reset"),
    ):
        with pytest.raises(requests.exceptions.RequestException):
            SerpexToolSpec(api_key="test-key").search("x")


# Integration test (requires real API key)
@pytest.mark.skipif(not os.environ.get("SERPEX_API_KEY"), reason="Requires real SERPEX API key")
def test_real_search():
    api_key = os.environ.get("SERPEX_API_KEY")
    tool = SerpexToolSpec(api_key=api_key)
    results = tool.search("LlamaIndex", num_results=5)
    assert 0 < len(results) <= 5
