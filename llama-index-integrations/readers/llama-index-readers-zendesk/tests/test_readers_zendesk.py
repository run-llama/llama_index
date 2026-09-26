from llama_index.core.readers.base import BaseReader
from llama_index.readers.zendesk import ZendeskReader


def test_class():
    names_of_base_classes = [b.__name__ for b in ZendeskReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_get_articles_page_sets_timeout(monkeypatch):
    import requests

    calls = []

    class _Response:
        text = '{"articles": [], "next_page": null}'

    def fake_get(url, **kwargs):
        calls.append(kwargs)
        return _Response()

    monkeypatch.setattr(requests, "get", fake_get)

    ZendeskReader("subdomain").get_articles_page()

    assert calls[0].get("timeout")
