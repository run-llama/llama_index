from llama_index.readers.stripe_docs import StripeDocsReader


def test_load_url_closes_response(monkeypatch):
    class Response:
        was_closed = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.was_closed = True

        def read(self):
            return b"<xml />"

    response = Response()
    monkeypatch.setattr("urllib.request.urlopen", lambda url: response)

    assert StripeDocsReader()._load_url("https://example.com/sitemap.xml") == b"<xml />"
    assert response.was_closed
