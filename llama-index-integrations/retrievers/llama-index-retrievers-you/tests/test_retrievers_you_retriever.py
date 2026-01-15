import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import QueryBundle
from llama_index.retrievers.you import YouRetriever


def test_class():
    names_of_base_classes = [b.__name__ for b in YouRetriever.__mro__]
    assert BaseRetriever.__name__ in names_of_base_classes


def test_api_key_from_argument():
    retriever = YouRetriever(api_key="test-key")
    assert retriever._api_key == "test-key"


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("YDC_API_KEY", "env-key")
    retriever = YouRetriever()
    assert retriever._api_key == "env-key"


def test_api_key_missing_raises():
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("YDC_API_KEY", None)
        with pytest.raises(ValueError, match="API key is required"):
            YouRetriever()


def test_generate_params_filters_none():
    retriever = YouRetriever(api_key="test", count=5, country="US")
    params = retriever._generate_params("test query")
    assert params == {"query": "test query", "count": 5, "country": "US"}
    assert "safesearch" not in params


def test_process_result_uses_snippets():
    retriever = YouRetriever(api_key="test")
    result = {
        "url": "https://example.com",
        "title": "Test",
        "snippets": ["snippet 1", "snippet 2"],
        "description": "fallback",
    }
    node = retriever._process_result(result, "web")
    assert node.text == "snippet 1\nsnippet 2"
    assert node.metadata["source_type"] == "web"


def test_process_result_falls_back_to_description():
    retriever = YouRetriever(api_key="test")
    result = {"url": "https://example.com", "description": "desc text"}
    node = retriever._process_result(result, "news")
    assert node.text == "desc text"


def test_process_result_handles_livecrawl_content():
    retriever = YouRetriever(api_key="test")
    result = {
        "url": "https://example.com",
        "snippets": ["text"],
        "contents": {"markdown": "# Title", "html": "<h1>Title</h1>"},
    }
    node = retriever._process_result(result, "web")
    assert node.metadata["content_markdown"] == "# Title"
    assert node.metadata["content_html"] == "<h1>Title</h1>"


def test_process_result_handles_none_contents():
    retriever = YouRetriever(api_key="test")
    result = {"url": "https://example.com", "contents": None}
    node = retriever._process_result(result, "web")
    assert "content_markdown" not in node.metadata


def test_process_result_filters_none_metadata():
    retriever = YouRetriever(api_key="test")
    result = {"url": "https://example.com", "title": None, "snippets": ["text"]}
    node = retriever._process_result(result, "web")
    assert "title" not in node.metadata
    assert node.metadata["url"] == "https://example.com"


@patch("llama_index.retrievers.you.base.requests.get")
def test_retrieve_processes_web_and_news(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": {
            "web": [{"url": "https://web.com", "snippets": ["web content"]}],
            "news": [{"url": "https://news.com", "description": "news content"}],
        }
    }
    mock_get.return_value = mock_response

    retriever = YouRetriever(api_key="test")
    results = retriever._retrieve(QueryBundle("test"))

    assert len(results) == 2
    assert results[0].node.metadata["source_type"] == "web"
    assert results[1].node.metadata["source_type"] == "news"
    assert all(r.score == 1.0 for r in results)


@patch("llama_index.retrievers.you.base.requests.get")
def test_retrieve_handles_empty_results(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {"results": {}}
    mock_get.return_value = mock_response

    retriever = YouRetriever(api_key="test")
    results = retriever._retrieve(QueryBundle("test"))
    assert results == []


@patch("llama_index.retrievers.you.base.requests.get")
def test_retrieve_timeout_raises(mock_get):
    mock_get.side_effect = requests.exceptions.Timeout("timeout")
    retriever = YouRetriever(api_key="test")
    with pytest.raises(ValueError, match="timed out"):
        retriever._retrieve(QueryBundle("test"))


@patch("llama_index.retrievers.you.base.requests.get")
def test_retrieve_request_error_raises(mock_get):
    mock_get.side_effect = requests.exceptions.ConnectionError("connection failed")
    retriever = YouRetriever(api_key="test")
    with pytest.raises(ValueError, match="request failed"):
        retriever._retrieve(QueryBundle("test"))


@patch("llama_index.retrievers.you.base.requests.get")
def test_retrieve_passes_correct_headers_and_params(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {"results": {}}
    mock_get.return_value = mock_response

    retriever = YouRetriever(api_key="my-key", count=10, country="US")
    retriever._retrieve(QueryBundle("search term"))

    mock_get.assert_called_once()
    call_kwargs = mock_get.call_args[1]
    assert call_kwargs["headers"]["X-API-Key"] == "my-key"
    assert call_kwargs["params"]["query"] == "search term"
    assert call_kwargs["params"]["count"] == 10
    assert call_kwargs["timeout"] == 30
