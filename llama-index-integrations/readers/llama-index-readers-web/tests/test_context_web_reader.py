import asyncio
import json
from copy import deepcopy

import httpx
import pytest

from llama_index.readers.web import ContextWebReader
from llama_index.readers.web.context_web.base import ContextAPIError


def _sync_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _async_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_reader_uses_environment_configuration(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_DEV_API_KEY", "environment-key")
    monkeypatch.setenv("CONTEXT_DEV_API_BASE", "https://context.example/v1/")

    reader = ContextWebReader()

    assert reader.api_key.get_secret_value() == "environment-key"
    assert reader.api_url == "https://context.example/v1"
    assert reader.mode == "scrape"
    assert reader.is_remote is True


def test_reader_supports_legacy_environment_configuration(monkeypatch) -> None:
    monkeypatch.delenv("CONTEXT_DEV_API_KEY", raising=False)
    monkeypatch.delenv("CONTEXT_DEV_API_BASE", raising=False)
    monkeypatch.setenv("CONTEXT_API_KEY", "legacy-key")
    monkeypatch.setenv("CONTEXT_API_BASE", "https://legacy.example/v1")

    reader = ContextWebReader()

    assert reader.api_key.get_secret_value() == "legacy-key"
    assert reader.api_url == "https://legacy.example/v1"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "API key missing"),
        ({"api_key": "   "}, "API key missing"),
        ({"api_key": "key", "api_url": "example.com"}, "must use HTTP"),
        ({"api_key": "key", "timeout": 0}, "greater than zero"),
    ],
)
def test_reader_rejects_invalid_configuration(kwargs, message, monkeypatch) -> None:
    monkeypatch.delenv("CONTEXT_API_KEY", raising=False)
    monkeypatch.delenv("CONTEXT_API_BASE", raising=False)
    monkeypatch.delenv("CONTEXT_DEV_API_KEY", raising=False)
    monkeypatch.delenv("CONTEXT_DEV_API_BASE", raising=False)

    with pytest.raises(ValueError, match=message):
        ContextWebReader(**kwargs)


@pytest.mark.parametrize(
    ("mode", "load_kwargs", "message"),
    [
        ("scrape", {}, "requires url"),
        ("crawl", {"query": "Context"}, "requires url"),
        ("sitemap", {}, "requires url"),
        ("search", {"url": "https://context.dev"}, "requires query"),
        ("extract", {"url": "https://context.dev"}, "JSON Schema"),
    ],
)
def test_reader_validates_mode_inputs(mode, load_kwargs, message) -> None:
    reader = ContextWebReader(api_key="key", mode=mode)

    with pytest.raises(ValueError, match=message):
        reader.load_data(**load_kwargs)


def test_scrape_builds_deep_query_and_returns_document() -> None:
    observed_request = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal observed_request
        observed_request = request
        return httpx.Response(
            200,
            json={
                "success": True,
                "markdown": "# Context.dev",
                "contentLength": 13,
                "url": "https://context.dev/docs",
                "metadata": {"title": "Context.dev Docs", "language": "en"},
                "key_metadata": {"credits_remaining": 10},
            },
        )

    constructor_params = {
        "includeLinks": True,
        "pdf": {"start": 2, "end": 4},
    }
    call_params = {"includeSelectors": ["main", "article"]}
    original_constructor_params = deepcopy(constructor_params)
    original_call_params = deepcopy(call_params)
    reader = ContextWebReader(
        api_key="secret",
        api_url="https://context.example/v1/",
        mode="scrape",
        params=constructor_params,
        client=_sync_client(handler),
    )

    documents = reader.load_data(
        url="https://context.dev/docs",
        params=call_params,
    )

    assert observed_request is not None
    assert observed_request.method == "GET"
    assert observed_request.url.path == "/v1/web/scrape/markdown"
    assert observed_request.url.params.get("includeLinks") == "true"
    assert observed_request.url.params.get("pdf[start]") == "2"
    assert observed_request.url.params.get("pdf[end]") == "4"
    assert observed_request.url.params.get_list("includeSelectors") == [
        "main",
        "article",
    ]
    assert observed_request.headers["Authorization"] == "Bearer secret"
    assert documents[0].text == "# Context.dev"
    assert documents[0].metadata == {
        "title": "Context.dev Docs",
        "language": "en",
        "content_length": 13,
        "url": "https://context.dev/docs",
        "source": "https://context.dev/docs",
        "mode": "scrape",
    }
    assert constructor_params == original_constructor_params
    assert call_params == original_call_params


def test_crawl_returns_one_document_per_page() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {
            "maxPages": 2,
            "url": "https://context.dev/docs",
        }
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "markdown": "# Docs",
                        "metadata": {
                            "url": "https://context.dev/docs",
                            "title": "Docs",
                            "crawlDepth": 0,
                            "statusCode": 200,
                            "success": True,
                        },
                    },
                    {
                        "markdown": "# Search",
                        "metadata": {
                            "url": "https://context.dev/docs/search",
                            "title": "Search",
                            "crawlDepth": 1,
                            "statusCode": 200,
                            "success": True,
                        },
                    },
                ],
                "metadata": {
                    "numUrls": 2,
                    "maxCrawlDepth": 1,
                    "numSucceeded": 2,
                    "numFailed": 0,
                    "numSkipped": 0,
                },
            },
        )

    reader = ContextWebReader(
        api_key="key",
        mode="crawl",
        params={"maxPages": 2},
        client=_sync_client(handler),
    )

    documents = reader.load_data(url="https://context.dev/docs")

    assert [document.text for document in documents] == ["# Docs", "# Search"]
    assert [document.metadata["crawlDepth"] for document in documents] == [0, 1]
    assert documents[1].metadata["source"] == "https://context.dev/docs/search"
    assert all(document.metadata["mode"] == "crawl" for document in documents)


def test_sitemap_returns_url_documents_with_crawl_metadata() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params.get("domain") == "context.dev"
        assert request.url.params.get("search") == "api authentication"
        return httpx.Response(
            200,
            json={
                "success": True,
                "domain": "context.dev",
                "urls": [
                    "https://context.dev/docs/auth",
                    "https://context.dev/docs/api",
                ],
                "meta": {
                    "sitemapsDiscovered": 1,
                    "sitemapsFetched": 1,
                    "sitemapsSkipped": 0,
                    "errors": 0,
                },
            },
        )

    reader = ContextWebReader(
        api_key="key",
        mode="sitemap",
        params={"search": "api authentication"},
        client=_sync_client(handler),
    )

    documents = reader.load_data(url="context.dev")

    assert [document.text for document in documents] == [
        "https://context.dev/docs/auth",
        "https://context.dev/docs/api",
    ]
    assert documents[0].metadata["domain"] == "context.dev"
    assert documents[0].metadata["sitemap"]["errors"] == 0
    assert documents[0].metadata["mode"] == "sitemap"


def test_search_prefers_markdown_and_falls_back_to_snippet() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {
            "numResults": 2,
            "markdownOptions": {"enabled": True},
            "query": "Context.dev launch",
        }
        return httpx.Response(
            200,
            json={
                "query": "Context.dev launch",
                "results": [
                    {
                        "url": "https://context.dev/blog/launch",
                        "title": "Launch",
                        "description": "Context.dev launched.",
                        "relevance": "high",
                        "markdown": {"markdown": "# Launch", "code": "SUCCESS"},
                    },
                    {
                        "url": "https://context.dev/about",
                        "title": "About",
                        "description": "About Context.dev.",
                        "relevance": "medium",
                        "markdown": {"markdown": None, "code": "NOT_REQUESTED"},
                    },
                    {
                        "url": "https://context.dev/changelog",
                        "title": None,
                        "description": None,
                        "relevance": "low",
                        "markdown": {"markdown": None, "code": "FAILED"},
                    },
                ],
            },
        )

    reader = ContextWebReader(
        api_key="key",
        mode="search",
        params={
            "numResults": 2,
            "markdownOptions": {"enabled": True},
        },
        client=_sync_client(handler),
    )

    documents = reader.load_data(query="Context.dev launch")

    assert documents[0].text == "# Launch"
    assert documents[0].metadata["markdown_status"] == "SUCCESS"
    assert documents[1].text == "About\n\nAbout Context.dev."
    assert documents[1].metadata["query"] == "Context.dev launch"
    assert documents[2].text == "https://context.dev/changelog"


def test_extract_returns_structured_json_document() -> None:
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {
            "schema": schema,
            "instructions": "Use the company name.",
            "url": "https://context.dev",
        }
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "url": "https://context.dev",
                "urls_analyzed": ["https://context.dev"],
                "data": {"name": "Context.dev"},
                "metadata": {
                    "numUrls": 1,
                    "maxCrawlDepth": 0,
                    "numSucceeded": 1,
                    "numFailed": 0,
                    "numSkipped": 0,
                    "numBlocked": 0,
                },
            },
        )

    reader = ContextWebReader(
        api_key="key",
        mode="extract",
        params={
            "schema": schema,
            "instructions": "Use the company name.",
        },
        client=_sync_client(handler),
    )

    documents = reader.load_data(url="https://context.dev")

    assert json.loads(documents[0].text) == {"name": "Context.dev"}
    assert documents[0].metadata["urls_analyzed"] == ["https://context.dev"]
    assert documents[0].metadata["extract"]["numSucceeded"] == 1


def test_metadata_callback_augments_protected_source_metadata() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "success": True,
                "markdown": "content",
                "contentLength": 7,
                "url": "https://context.dev",
                "metadata": {},
            },
        )

    reader = ContextWebReader(
        api_key="key",
        metadata_fn=lambda url: {"tenant": "demo", "url": "overridden"},
        client=_sync_client(handler),
    )

    document = reader.load_data(url="https://context.dev")[0]

    assert document.metadata["tenant"] == "demo"
    assert document.metadata["url"] == "https://context.dev"
    assert document.metadata["source"] == "https://context.dev"


def test_api_errors_include_status_code_and_error_code() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={"message": "Credits exhausted", "error_code": "USAGE_EXCEEDED"},
        )

    reader = ContextWebReader(api_key="key", client=_sync_client(handler))

    with pytest.raises(ContextAPIError, match="403.*USAGE_EXCEEDED.*Credits") as exc:
        reader.load_data(url="https://context.dev")

    assert exc.value.status_code == 403
    assert exc.value.error_code == "USAGE_EXCEEDED"


def test_invalid_success_response_raises_clear_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    reader = ContextWebReader(api_key="key", client=_sync_client(handler))

    with pytest.raises(RuntimeError, match="invalid JSON"):
        reader.load_data(url="https://context.dev")


def test_async_reader_uses_async_transport() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        return httpx.Response(
            200,
            json={
                "query": "Context.dev",
                "results": [
                    {
                        "url": "https://context.dev",
                        "title": "Context.dev",
                        "description": "Web infrastructure for AI.",
                        "relevance": "high",
                        "markdown": {"markdown": None, "code": "NOT_REQUESTED"},
                    }
                ],
            },
        )

    reader = ContextWebReader(
        api_key="key",
        mode="search",
        async_client=_async_client(handler),
    )

    documents = asyncio.run(reader.aload_data(query="Context.dev"))

    assert documents[0].metadata["url"] == "https://context.dev"
    assert documents[0].metadata["mode"] == "search"
