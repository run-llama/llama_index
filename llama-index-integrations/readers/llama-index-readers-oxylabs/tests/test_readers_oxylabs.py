import os
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

import pytest
from llama_index.core.readers.base import BaseReader
from llama_index.readers.oxylabs.base import OxylabsBaseReader
from llama_index.readers.oxylabs import (
    OxylabsAmazonBestsellersReader,
    OxylabsAmazonPricingReader,
    OxylabsAmazonProductReader,
    OxylabsAmazonReviewsReader,
    OxylabsAmazonSellersReader,
    OxylabsAmazonSearchReader,
    OxylabsGoogleAdsReader,
    OxylabsGoogleSearchReader,
    OxylabsYoutubeTranscriptReader,
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


READER_TESTS_PARAMS = [
    pytest.param(
        OxylabsAmazonBestsellersReader,
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
        {
            "query": "B087TXHLVQ",
            "parse": True,
        },
        id="amazon_pricing_successful_response",
    ),
    pytest.param(
        OxylabsAmazonProductReader,
        {
            "query": "B087TXHLVQ",
            "parse": True,
        },
        id="amazon_product_successful_response",
    ),
    pytest.param(
        OxylabsAmazonReviewsReader,
        {
            "query": "B087TXHLVQ",
            "parse": True,
        },
        id="amazon_reviews_successful_response",
    ),
    pytest.param(
        OxylabsAmazonSellersReader,
        {
            "query": "A2U55XLSPNCN01",
            "parse": True,
        },
        id="amazon_sellers_successful_response",
    ),
    pytest.param(
        OxylabsAmazonSearchReader,
        {
            "query": "headsets",
            "parse": True,
        },
        id="amazon_search_successful_response",
    ),
    pytest.param(
        OxylabsYoutubeTranscriptReader,
        {
            "query": "SLoqvcnwwN4",
            "context": [
                {"key": "language_code", "value": "en"},
                {"key": "transcript_origin", "value": "uploader_provided"},
            ],
        },
        id="youtube_transcript_response",
    ),
]


@pytest.mark.parametrize(
    ("reader_class", "payload"),
    READER_TESTS_PARAMS,
)
@pytest.mark.unit
def test_sync_oxylabs_readers(
    reader_class: type[OxylabsBaseReader],
    payload: dict,
):
    get_response_mock = MagicMock()
    get_response_mock.return_value = get_response()
    reader_class.get_response = get_response_mock

    reader = reader_class(
        username="OXYLABS_USERNAME",
        password="OXYLABS_PASSWORD",
    )

    docs = reader.load_data(payload)

    assert (
        docs[0].text == f"# {reader.top_level_header}\n"
        f"- Item 1:\n  ## key1\n    value1\n\n  ## key2\n    value2\n"
    )

    assert get_response_mock.call_args[0][0] == payload


@pytest.mark.parametrize(
    ("reader_class", "payload"),
    READER_TESTS_PARAMS,
)
@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_oxylabs_readers(
    reader_class: type[OxylabsBaseReader],
    payload: dict,
):
    get_response_mock = AsyncMock()
    get_response_mock.return_value = get_response()
    reader_class.aget_response = get_response_mock

    reader = reader_class(
        username="OXYLABS_USERNAME",
        password="OXYLABS_PASSWORD",
    )

    docs = await reader.aload_data(payload)

    assert (
        docs[0].text == f"# {reader.top_level_header}\n"
        f"- Item 1:\n  ## key1\n    value1\n\n  ## key2\n    value2\n"
    )

    assert get_response_mock.call_args[0][0] == payload


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


@pytest.mark.skipif(
    not (os.environ.get("OXYLABS_USERNAME") and os.environ.get("OXYLABS_PASSWORD")),
    reason="No Oxylabs creds",
)
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


@pytest.mark.skipif(
    not (os.environ.get("OXYLABS_USERNAME") and os.environ.get("OXYLABS_PASSWORD")),
    reason="No Oxylabs creds",
)
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
