import json
import os
from typing import Any

import pytest

from llama_index.readers.web import ContextWebReader
from llama_index.readers.web.context_web.base import ContextMode


def _api_key() -> str:
    api_key = (
        os.getenv("CONTEXT_DEV_API_KEY") or os.getenv("CONTEXT_API_KEY", "")
    ).strip()
    if not api_key:
        pytest.skip("CONTEXT_DEV_API_KEY not set")
    return api_key


def _reader(mode: ContextMode, **params: Any) -> ContextWebReader:
    return ContextWebReader(
        api_key=_api_key(),
        api_url=os.getenv("CONTEXT_DEV_API_BASE") or os.getenv("CONTEXT_API_BASE"),
        mode=mode,
        params=params,
    )


@pytest.mark.integration
def test_live_scrape() -> None:
    documents = _reader("scrape", useMainContentOnly=True).load_data(
        url="https://www.context.dev"
    )

    assert len(documents) == 1
    assert documents[0].text.strip()
    assert documents[0].metadata["url"].startswith("https://")


@pytest.mark.integration
def test_live_crawl() -> None:
    documents = _reader("crawl", maxPages=2, maxDepth=1).load_data(
        url="https://docs.context.dev"
    )

    assert 1 <= len(documents) <= 2
    assert all(document.text.strip() for document in documents)
    assert all(document.metadata["mode"] == "crawl" for document in documents)


@pytest.mark.integration
def test_live_sitemap() -> None:
    documents = _reader("sitemap", maxLinks=3).load_data(url="context.dev")

    assert 1 <= len(documents) <= 3
    assert all(document.text.startswith("http") for document in documents)
    assert all(document.metadata["mode"] == "sitemap" for document in documents)


@pytest.mark.integration
def test_live_search() -> None:
    documents = _reader(
        "search",
        numResults=10,
        includeDomains=["context.dev"],
        markdownOptions={"enabled": True},
    ).load_data(query="Context.dev web scraping API")

    assert documents
    assert all(document.text.strip() for document in documents)
    assert all(document.metadata["query"] for document in documents)


@pytest.mark.integration
def test_live_extract() -> None:
    documents = _reader(
        "extract",
        schema={
            "type": "object",
            "properties": {"company_name": {"type": "string"}},
            "required": ["company_name"],
        },
        instructions="Extract the company name.",
        maxPages=1,
        maxDepth=0,
    ).load_data(url="https://www.context.dev")

    assert len(documents) == 1
    assert json.loads(documents[0].text)["company_name"]
    assert documents[0].metadata["mode"] == "extract"
