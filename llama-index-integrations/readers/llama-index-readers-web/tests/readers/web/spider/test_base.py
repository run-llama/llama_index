from llama_index.readers.web.spider.base import SpiderWebReader


def test_spider_base_functionality():
    reader = SpiderWebReader(api_key="fake_api_key")
    result = reader.load_data("https://example.com")
    assert result is not None
    assert result == isinstance(list)
