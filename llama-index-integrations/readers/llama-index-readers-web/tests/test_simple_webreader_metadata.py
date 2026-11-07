import pytest
import requests

from llama_index.readers.web import SimpleWebPageReader


@pytest.fixture()
def web_response(monkeypatch):
    def get(url, **kwargs):
        response = requests.Response()
        response.status_code = 200
        response.encoding = "utf-8"
        response._content = b"<p>Page content</p>"
        return response

    monkeypatch.setattr(requests, "get", get)


@pytest.mark.parametrize("html_to_text", [False, True])
def test_shared_callback_metadata_keeps_each_page_url(web_response, html_to_text):
    metadata = {"source": "docs"}
    urls = ["https://example.test/first", "https://example.test/second"]
    reader = SimpleWebPageReader(
        metadata_fn=lambda url: metadata, html_to_text=html_to_text
    )

    documents = reader.load_data(urls)

    assert [doc.metadata for doc in documents] == [
        {"source": "docs", "url": url} for url in urls
    ]
    assert all("Page content" in doc.text for doc in documents)


def test_callback_metadata_is_not_modified(web_response):
    metadata = {"source": "docs"}
    reader = SimpleWebPageReader(metadata_fn=lambda url: metadata)

    reader.load_data(["https://example.test/page"])

    assert metadata == {"source": "docs"}


def test_callback_can_override_url(web_response):
    metadata = {"source": "docs", "url": "https://example.test/canonical"}
    reader = SimpleWebPageReader(metadata_fn=lambda url: metadata)

    documents = reader.load_data(["https://example.test/page"])

    assert documents[0].metadata == metadata
    assert documents[0].metadata["url"] == "https://example.test/canonical"


def test_default_metadata_keeps_page_url(web_response):
    url = "https://example.test/page"

    documents = SimpleWebPageReader().load_data([url])

    assert documents[0].metadata == {"url": url}
