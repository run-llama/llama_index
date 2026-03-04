import os

import pytest
from typing import Optional

from llama_index.readers.web.firecrawl_web.base import FireCrawlWebReader

# Set PRINT_RESULTS = True to print documents, otherwise tests will use asserts.
PRINT_RESULTS = False


@pytest.fixture(scope="session", autouse=True)
def _print_firecrawl_version() -> None:
    try:
        import firecrawl  # type: ignore

        version = getattr(firecrawl, "__version__", None)
        print(f"firecrawl version: {version}")
    except Exception as exc:
        print(f"firecrawl import failed: {exc}")


def _require_api_key() -> str:
    api_key = os.getenv("FIRECRAWL_API_KEY", "").strip()
    if not api_key:
        pytest.skip("FIRECRAWL_API_KEY not set")
    return api_key


def _api_url() -> Optional[str]:
    return os.getenv("FIRECRAWL_API_URL") or os.getenv("FIRECRAWL_BASE_URL")


TEST_URL = os.getenv("FIRECRAWL_TEST_URL", "https://example.pt")
TEST_QUERY = os.getenv("FIRECRAWL_TEST_QUERY", "LlamaIndex")
TEST_PROMPT = os.getenv("FIRECRAWL_TEST_PROMPT", "Extract the title as 'title'")


def test_scrape_prints_documents() -> None:
    reader = FireCrawlWebReader(
        api_key=_require_api_key(),
        api_url=_api_url(),
        mode="scrape",
        params={"formats": ["markdown"]},
    )
    for doc in reader.load_data(url=TEST_URL):
        if PRINT_RESULTS:
            print(f"[SCRAPE] document: {doc}")
        else:
            assert doc.text is not None
            assert doc.metadata is not None


def test_crawl_prints_documents() -> None:
    reader = FireCrawlWebReader(
        api_key=_require_api_key(),
        api_url=_api_url(),
        mode="crawl",
        params={"limit": 3},
    )
    for doc in reader.load_data(url=TEST_URL):
        if PRINT_RESULTS:
            print(f"[CRAWL] document: {doc}")
        else:
            assert doc.text is not None
            assert doc.metadata is not None


def test_map_prints_documents() -> None:
    reader = FireCrawlWebReader(
        api_key=_require_api_key(),
        api_url=_api_url(),
        mode="map",
        params={"limit": 10},
    )
    for doc in reader.load_data(url=TEST_URL):
        if PRINT_RESULTS:
            print(f"[MAP] document: {doc}")
        else:
            assert doc.text is not None
            assert doc.metadata is not None


def test_search_prints_documents() -> None:
    reader = FireCrawlWebReader(
        api_key=_require_api_key(),
        api_url=_api_url(),
        mode="search",
    )
    for doc in reader.load_data(query=TEST_QUERY):
        if PRINT_RESULTS:
            print(f"[SEARCH] document: {doc}")
        else:
            assert doc.text is not None
            assert doc.metadata is not None


def test_extract_prints_documents() -> None:
    reader = FireCrawlWebReader(
        api_key=_require_api_key(),
        api_url=_api_url(),
        mode="extract",
        params={"prompt": TEST_PROMPT},
    )
    for doc in reader.load_data(urls=[TEST_URL]):
        if PRINT_RESULTS:
            print(f"[EXTRACT] document: {doc}")
        else:
            assert doc.text is not None
            assert doc.metadata is not None
