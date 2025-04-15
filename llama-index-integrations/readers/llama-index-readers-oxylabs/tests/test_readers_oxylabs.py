import asyncio
import os

from llama_index.core.readers.base import BaseReader
from llama_index.readers.oxylabs.base import OxylabsBaseReader


def test_class():
    names_of_base_classes = [b.__name__ for b in OxylabsBaseReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_google_search_reader():
    from llama_index.readers.oxylabs import OxylabsGoogleSearchReader

    reader = OxylabsGoogleSearchReader(
        username=os.environ["OXYLABS_USERNAME"], password=os.environ["OXYLABS_PASSWORD"]
    )

    docs = reader.load_data(
        {
            "parse": True,
            "query": "Iphone 16",
            "geo_location": "Paris, France",
        }
    )

    print(docs[0].text)


def test_google_ads_reader():
    from llama_index.readers.oxylabs import OxylabsGoogleAdsReader

    reader = OxylabsGoogleAdsReader(
        username=os.environ["OXYLABS_USERNAME"], password=os.environ["OXYLABS_PASSWORD"]
    )

    docs = reader.load_data(
        {
            "parse": True,
            "query": "Iphone 16",
            "geo_location": "Paris, France",
        }
    )

    print(docs[0].text)


def test_amazon_search_reader():
    from llama_index.readers.oxylabs.amazon_search import (
        OxylabsAmazonSearchReader,
    )

    reader = OxylabsAmazonSearchReader(
        username=os.environ["OXYLABS_USERNAME"], password=os.environ["OXYLABS_PASSWORD"]
    )

    docs = asyncio.run(
        reader.aload_data(
            {
                "query": "headsets",
                "parse": True,
            }
        )
    )

    print(docs[0].text)


def test_amazon_bestsellers_reader():
    from llama_index.readers.oxylabs import (
        OxylabsAmazonBestsellersReader,
    )

    reader = OxylabsAmazonBestsellersReader(
        username=os.environ["OXYLABS_USERNAME"], password=os.environ["OXYLABS_PASSWORD"]
    )

    docs = reader.load_data(
        {
            "source": "amazon_bestsellers",
            "domain": "com",
            "query": "120225786011",
            "render": "html",
            "start_page": 1,
            "parse": True,
        }
    )

    print(docs[0].text)


def test_amazon_pricing_reader():
    from llama_index.readers.oxylabs import (
        OxylabsAmazonPricingReader,
    )

    reader = OxylabsAmazonPricingReader(
        username=os.environ["OXYLABS_USERNAME"], password=os.environ["OXYLABS_PASSWORD"]
    )

    docs = reader.load_data(
        {
            "query": "B087TXHLVQ",
            "parse": True,
        }
    )

    print(docs[0].text)


def test_amazon_product_reader():
    from llama_index.readers.oxylabs import (
        OxylabsAmazonProductReader,
    )

    reader = OxylabsAmazonProductReader(
        username=os.environ["OXYLABS_USERNAME"], password=os.environ["OXYLABS_PASSWORD"]
    )

    docs = reader.load_data(
        {
            "query": "B087TXHLVQ",
            "parse": True,
        }
    )

    print(docs[0].text)


def test_amazon_reviews_reader():
    from llama_index.readers.oxylabs import (
        OxylabsAmazonReviewsReader,
    )

    reader = OxylabsAmazonReviewsReader(
        username=os.environ["OXYLABS_USERNAME"], password=os.environ["OXYLABS_PASSWORD"]
    )

    docs = reader.load_data(
        {
            "query": "B087TXHLVQ",
            "parse": True,
        }
    )

    print(docs[0].text)


def test_amazon_sellers_reader():
    from llama_index.readers.oxylabs import (
        OxylabsAmazonSellersReader,
    )

    reader = OxylabsAmazonSellersReader(
        username=os.environ["OXYLABS_USERNAME"], password=os.environ["OXYLABS_PASSWORD"]
    )

    docs = reader.load_data(
        {
            "query": "A2U55XLSPNCN01",
            "parse": True,
        }
    )

    print(docs[0].text)


def test_youtube_transcripts_reader():
    from llama_index.readers.oxylabs import (
        OxylabsYoutubeTranscriptsReader,
    )

    reader = OxylabsYoutubeTranscriptsReader(
        username=os.environ["OXYLABS_USERNAME"], password=os.environ["OXYLABS_PASSWORD"]
    )

    docs = reader.load_data(
        {
            "query": "SLoqvcnwwN4",
            "context": [
                {"key": "language_code", "value": "en"},
                {"key": "transcript_origin", "value": "uploader_provided"},
            ],
        }
    )

    print(docs[0].text)
