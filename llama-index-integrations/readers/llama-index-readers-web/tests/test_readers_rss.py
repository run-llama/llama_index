import os

from llama_index.readers.web import RssReader


def test_rss_reader_non_strict_sources():
    default_reader = RssReader()
    documents = default_reader.load_data(urls=["https://news.ycombinator.com/rss"])
    assert len(documents) > 0


def test_rss_reader_rsshub():
    default_reader = RssReader()
    documents = default_reader.load_data(urls=["https://rsshub.app/hackernews/newest"])
    assert len(documents) == 0


def test_rss_reader_user_agent():
    reader = RssReader(user_agent="MyApp/1.0 +http://example.com/")
    documents = reader.load_data(urls=["https://rsshub.app/hackernews/newest"])
    assert len(documents) > 0


def test_oxylabs():
    from llama_index.readers.web.oxylabs_web.base import OxylabsWebReader

    reader = OxylabsWebReader(
        username=os.environ["OXYLABS_USERNAME"], password=os.environ["OXYLABS_PASSWORD"]
    )

    docs = reader.load_data(
        [
            "https://sandbox.oxylabs.io/products/1",
            "https://sandbox.oxylabs.io/products/2",
        ],
        {
            "parse": False,
        },
    )

    print(docs[0].text)
