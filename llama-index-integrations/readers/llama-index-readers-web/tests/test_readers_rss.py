from llama_index.readers.web import RssReader


def test_rss_reader_non_strict_sources():
    default_reader = RssReader()
    documents = default_reader.load_data(urls=["https://news.ycombinator.com/rss"])
    assert len(documents) > 0


def test_rss_reader_user_agent():
    reader = RssReader(user_agent="MyApp/1.0 +http://example.com/")
    documents = reader.load_data(urls=["https://news.ycombinator.com/rss"])
    assert len(documents) > 0
