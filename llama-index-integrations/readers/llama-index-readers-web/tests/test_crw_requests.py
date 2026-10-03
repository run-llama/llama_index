import os

import pytest
from typing import Optional

from llama_index.readers.web.crw_web.base import CrwWebReader

# Set PRINT_RESULTS = True to print documents, otherwise tests will use asserts.
PRINT_RESULTS = False


@pytest.fixture(scope="session", autouse=True)
def _print_crw_version() -> None:
    try:
        import crw  # type: ignore

        version = getattr(crw, "__version__", None)
        print(f"crw version: {version}")
    except Exception as exc:
        print(f"crw import failed: {exc}")


def _require_api_key() -> str:
    api_key = os.getenv("CRW_API_KEY", "").strip()
    if not api_key:
        pytest.skip("CRW_API_KEY not set")
    return api_key


def _api_url() -> Optional[str]:
    return os.getenv("CRW_API_URL") or os.getenv("CRW_BASE_URL")


TEST_URL = os.getenv("CRW_TEST_URL", "https://example.pt")
TEST_QUERY = os.getenv("CRW_TEST_QUERY", "LlamaIndex")


def test_scrape_prints_documents() -> None:
    reader = CrwWebReader(
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
    reader = CrwWebReader(
        api_key=_require_api_key(),
        api_url=_api_url(),
        mode="crawl",
        params={"maxPages": 3},
    )
    for doc in reader.load_data(url=TEST_URL):
        if PRINT_RESULTS:
            print(f"[CRAWL] document: {doc}")
        else:
            assert doc.text is not None
            assert doc.metadata is not None


def test_map_prints_documents() -> None:
    reader = CrwWebReader(
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
    reader = CrwWebReader(
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
