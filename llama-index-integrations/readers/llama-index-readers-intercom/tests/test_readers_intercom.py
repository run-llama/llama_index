from llama_index.core.readers.base import BaseReader
from llama_index.readers.intercom import IntercomReader


def test_class():
    names_of_base_classes = [b.__name__ for b in IntercomReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_get_articles_page_sets_timeout(monkeypatch):
    import requests
    from llama_index.readers.intercom import IntercomReader

    calls = []

    class _Response:
        text = '{"data": [], "pages": {}}'

    def fake_get(url, **kwargs):
        calls.append(kwargs)
        return _Response()

    monkeypatch.setattr(requests, "get", fake_get)

    IntercomReader("token").get_articles_page()

    assert calls[0].get("timeout")
