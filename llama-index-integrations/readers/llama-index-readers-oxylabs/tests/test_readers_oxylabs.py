import os
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

import pytest
from llama_index.core.readers.base import BaseReader
from llama_index.readers.oxylabs.base import OxylabsBaseReader
from llama_index.readers.oxylabs import (
    RESULT_CATEGORIES,
    OxylabsAmazonBestsellersReader,
    OxylabsAmazonPricingReader,
    OxylabsAmazonProductReader,
    OxylabsAmazonSellersReader,
    OxylabsAmazonSearchReader,
    OxylabsGoogleAdsReader,
    OxylabsGoogleSearchReader,
)

TEST_ROOT = Path(__file__).parent.resolve()


def test_class():
    names_of_base_classes = [b.__name__ for b in OxylabsBaseReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def get_response() -> MagicMock:
    mock = MagicMock()
    mock.raw = {
        "results": [{"content": {"key1": "value1", "key2": "value2"}}],
        "job": {"job_id": 42424242},
    }
    return mock


# `source_attr` and `method_name` are the Oxylabs SDK client coordinates each
# reader depends on. Mocking here rather than on the reader's own
# `get_response` keeps the SDK call path under test, so an SDK rename or a
# retired source fails these tests instead of only failing in production.
READER_TESTS_PARAMS = [
    pytest.param(
        OxylabsAmazonBestsellersReader,
        "amazon",
        "scrape_bestsellers",
        {
            "source": "amazon_bestsellers",
            "domain": "com",
            "query": "120225786011",
            "render": "html",
            "start_page": 1,
            "parse": True,
        },
        id="amazon_bestsellers_successful_response",
    ),
    pytest.param(
        OxylabsAmazonPricingReader,
        "amazon",
        "scrape_pricing",
        {
            "query": "B087TXHLVQ",
            "parse": True,
        },
        id="amazon_pricing_successful_response",
    ),
    pytest.param(
        OxylabsAmazonProductReader,
        "amazon",
        "scrape_product",
        {
            "query": "B087TXHLVQ",
            "parse": True,
        },
        id="amazon_product_successful_response",
    ),
    pytest.param(
        OxylabsAmazonSellersReader,
        "amazon",
        "scrape_sellers",
        {
            "query": "A2U55XLSPNCN01",
            "parse": True,
        },
        id="amazon_sellers_successful_response",
    ),
    pytest.param(
        OxylabsAmazonSearchReader,
        "amazon",
        "scrape_search",
        {
            "query": "headsets",
            "parse": True,
        },
        id="amazon_search_successful_response",
    ),
]


@pytest.mark.parametrize(
    ("reader_class", "source_attr", "method_name", "payload"),
    READER_TESTS_PARAMS,
)
@pytest.mark.unit
def test_sync_oxylabs_readers(
    reader_class: type[OxylabsBaseReader],
    source_attr: str,
    method_name: str,
    payload: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    reader = reader_class(
        username="OXYLABS_USERNAME",
        password="OXYLABS_PASSWORD",
    )

    # Resolving these off the real client asserts the SDK still exposes them.
    source = getattr(reader.oxylabs_api, source_attr)
    assert hasattr(source, method_name)

    scrape_mock = MagicMock(return_value=get_response())
    monkeypatch.setattr(source, method_name, scrape_mock)

    docs = reader.load_data(payload)

    assert (
        docs[0].text == f"# {reader.top_level_header}\n"
        f"- Item 1:\n  ## key1\n    value1\n\n  ## key2\n    value2\n"
    )

    scrape_mock.assert_called_once_with(**payload)


@pytest.mark.parametrize(
    ("reader_class", "source_attr", "method_name", "payload"),
    READER_TESTS_PARAMS,
)
@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_oxylabs_readers(
    reader_class: type[OxylabsBaseReader],
    source_attr: str,
    method_name: str,
    payload: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    reader = reader_class(
        username="OXYLABS_USERNAME",
        password="OXYLABS_PASSWORD",
    )

    source = getattr(reader.async_oxylabs_api, source_attr)
    assert hasattr(source, method_name)

    scrape_mock = AsyncMock(return_value=get_response())
    monkeypatch.setattr(source, method_name, scrape_mock)

    docs = await reader.aload_data(payload)

    assert (
        docs[0].text == f"# {reader.top_level_header}\n"
        f"- Item 1:\n  ## key1\n    value1\n\n  ## key2\n    value2\n"
    )

    scrape_mock.assert_called_once_with(**payload)


@pytest.mark.unit
def test_readers_are_marked_remote():
    reader = OxylabsAmazonSearchReader(
        username="OXYLABS_USERNAME",
        password="OXYLABS_PASSWORD",
    )
    assert reader.is_remote is True


@pytest.mark.unit
def test_api_error_message_is_surfaced(monkeypatch: pytest.MonkeyPatch):
    """A rejected job should report the API's reason, not a KeyError."""
    reader = OxylabsAmazonSearchReader(
        username="OXYLABS_USERNAME",
        password="OXYLABS_PASSWORD",
    )

    rejected = MagicMock()
    rejected.raw = {"message": "Unsupported source."}
    monkeypatch.setattr(
        reader.oxylabs_api.amazon, "scrape_search", MagicMock(return_value=rejected)
    )

    with pytest.raises(RuntimeError, match="Unsupported source."):
        reader.load_data({"query": "headsets", "parse": True})


GOOGLE_READER_TESTS_PARAMS = [
    pytest.param(
        OxylabsGoogleSearchReader,
        "google_search",
        {
            "query": "iPhone 16",
            "parse": True,
        },
        id="google_search_successful_response",
    ),
    pytest.param(
        OxylabsGoogleAdsReader,
        "google_ads",
        {
            "query": "iPhone 16",
            "parse": True,
        },
        id="google_ads_successful_response",
    ),
]

INTEGRATION_READER_PARAMS = [
    *GOOGLE_READER_TESTS_PARAMS,
    *[
        pytest.param(param.values[0], param.values[2], param.values[3], id=param.id)
        for param in READER_TESTS_PARAMS
    ],
]

requires_credentials = pytest.mark.skipif(
    not (os.environ.get("OXYLABS_USERNAME") and os.environ.get("OXYLABS_PASSWORD")),
    reason="No Oxylabs creds",
)


@requires_credentials
@pytest.mark.parametrize(
    ("reader_class", "name", "payload"),
    INTEGRATION_READER_PARAMS,
)
@pytest.mark.integration
def test_sync_oxylabs_readers_live(
    reader_class: type[OxylabsBaseReader],
    name: str,
    payload: dict,
):
    """
    Call the live API for every reader.

    Guards against a source being retired or an SDK namespace being renamed
    underneath the integration, neither of which the unit tests can observe.
    """
    reader = reader_class(
        username=os.environ.get("OXYLABS_USERNAME"),
        password=os.environ.get("OXYLABS_PASSWORD"),
    )

    docs = reader.load_data(payload)

    assert len(docs) == 1
    assert docs[0].text.strip()


@requires_credentials
@pytest.mark.parametrize(
    ("reader_class", "name", "payload"),
    INTEGRATION_READER_PARAMS,
)
@pytest.mark.asyncio
@pytest.mark.integration
async def test_async_oxylabs_readers_live(
    reader_class: type[OxylabsBaseReader],
    name: str,
    payload: dict,
):
    reader = reader_class(
        username=os.environ.get("OXYLABS_USERNAME"),
        password=os.environ.get("OXYLABS_PASSWORD"),
    )

    docs = await reader.aload_data(payload)

    assert len(docs) == 1
    assert docs[0].text.strip()


@requires_credentials
@pytest.mark.parametrize(
    ("reader_class", "name", "payload"),
    GOOGLE_READER_TESTS_PARAMS,
)
@pytest.mark.integration
def test_sync_google_oxylabs_readers(
    reader_class: type[OxylabsBaseReader],
    name: str,
    payload: dict,
):
    reader = reader_class(
        username=os.environ.get("OXYLABS_USERNAME"),
        password=os.environ.get("OXYLABS_PASSWORD"),
    )

    docs = reader.load_data(payload)

    assert len(docs) == 1

    text = docs[0].text

    assert len(text) > 1000
    assert "ORGANIC RESULTS ITEMS" in text
    assert "SEARCH INFORMATION" in text
    assert "RELATED SEARCHES ITEMS" in text


@requires_credentials
@pytest.mark.parametrize(
    ("reader_class", "name", "payload"),
    GOOGLE_READER_TESTS_PARAMS,
)
@pytest.mark.asyncio
@pytest.mark.integration
async def test_async_google_oxylabs_readers(
    reader_class: type[OxylabsBaseReader],
    name: str,
    payload: dict,
):
    reader = reader_class(
        username=os.environ.get("OXYLABS_USERNAME"),
        password=os.environ.get("OXYLABS_PASSWORD"),
    )

    docs = await reader.aload_data(payload)

    assert len(docs) == 1

    text = docs[0].text

    assert len(text) > 1000
    assert "ORGANIC RESULTS ITEMS" in text
    assert "SEARCH INFORMATION" in text
    assert "RELATED SEARCHES ITEMS" in text


GOOGLE_RESPONSE_RAW = {
    "results": [
        {
            "content": {
                "results": {
                    "organic": [{"title": "an organic result"}],
                    "search_information": {"query": "iPhone 16"},
                    "knowledge": {"title": "a knowledge graph entry"},
                }
            }
        }
    ],
    "job": {"job_id": 42424242},
}


def google_response() -> MagicMock:
    mock = MagicMock()
    mock.raw = GOOGLE_RESPONSE_RAW
    return mock


@pytest.mark.unit
def test_result_categories_defaults_to_every_category(monkeypatch: pytest.MonkeyPatch):
    reader = OxylabsGoogleSearchReader(
        username="OXYLABS_USERNAME",
        password="OXYLABS_PASSWORD",
    )
    assert reader.result_categories == []

    monkeypatch.setattr(
        reader.oxylabs_api.google,
        "scrape_search",
        MagicMock(return_value=google_response()),
    )

    text = reader.load_data({"query": "iPhone 16", "parse": True})[0].text

    assert "ORGANIC RESULTS ITEMS" in text
    assert "SEARCH INFORMATION" in text
    assert "KNOWLEDGE GRAPH" in text


@pytest.mark.unit
def test_result_categories_filters_output(monkeypatch: pytest.MonkeyPatch):
    reader = OxylabsGoogleSearchReader(
        username="OXYLABS_USERNAME",
        password="OXYLABS_PASSWORD",
        result_categories=["search_information"],
    )

    monkeypatch.setattr(
        reader.oxylabs_api.google,
        "scrape_search",
        MagicMock(return_value=google_response()),
    )

    text = reader.load_data({"query": "iPhone 16", "parse": True})[0].text

    assert "SEARCH INFORMATION" in text
    assert "ORGANIC RESULTS ITEMS" not in text
    assert "KNOWLEDGE GRAPH" not in text


@pytest.mark.unit
def test_unknown_result_category_is_rejected():
    """A typo must fail loudly, not silently return every category."""
    with pytest.raises(ValueError, match="knowledge_grahp"):
        OxylabsGoogleSearchReader(
            username="OXYLABS_USERNAME",
            password="OXYLABS_PASSWORD",
            result_categories=["knowledge_grahp"],
        )


@pytest.mark.unit
def test_every_advertised_category_is_accepted():
    reader = OxylabsGoogleSearchReader(
        username="OXYLABS_USERNAME",
        password="OXYLABS_PASSWORD",
        result_categories=RESULT_CATEGORIES,
    )
    assert reader.result_categories == RESULT_CATEGORIES


@requires_credentials
@pytest.mark.integration
def test_result_categories_shrinks_live_document():
    """Filtering should measurably reduce the document sent downstream."""
    payload = {"query": "iPhone 16", "parse": True}
    credentials = {
        "username": os.environ.get("OXYLABS_USERNAME"),
        "password": os.environ.get("OXYLABS_PASSWORD"),
    }

    full = OxylabsGoogleSearchReader(**credentials).load_data(payload)[0].text
    filtered = (
        OxylabsGoogleSearchReader(
            **credentials, result_categories=["search_information"]
        )
        .load_data(payload)[0]
        .text
    )

    assert "SEARCH INFORMATION" in filtered
    assert "ORGANIC RESULTS ITEMS" not in filtered
    assert len(filtered) < len(full)
