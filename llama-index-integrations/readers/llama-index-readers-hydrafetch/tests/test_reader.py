from __future__ import annotations

from typing import Any

import pytest
from llama_index.core.readers.base import BaseReader

from llama_index.readers.hydrafetch import HydrafetchReader

SCRAPE = {
    "data": {
        "url": "https://example.com",
        "finalUrl": "https://example.com/",
        "status": 200,
        "markdown": "# Example\n\nBody text.",
        "metadata": {
            "title": "Example",
            "description": "An example page.",
            "language": "en",
            "siteName": "Example",
            "wordCount": 3,
            "pageType": "article",
        },
    }
}

NOT_FOUND = {"data": {"url": "https://example.com/gone", "status": 404, "markdown": "# Not Found"}}

MAP = {
    "data": {
        "links": [
            {"url": "https://example.com/a", "lastmod": "2026-08-01"},
            {"url": "https://example.com/b"},
            {"nope": True},
        ]
    }
}

CRAWL = {
    "data": {
        "status": "completed",
        "pages": [
            {
                "url": "https://ex.com/a",
                "depth": 1,
                "data": {"url": "a", "status": 200, "markdown": "A"},
            },
            {"url": "https://ex.com/b", "data": {"url": "b", "status": 404, "markdown": "B"}},
            {"url": "https://ex.com/c", "data": None},
        ],
    }
}


class FakeClient:
    def __init__(self, **payloads: Any) -> None:
        self.payloads = payloads
        self.calls: list[tuple[str, Any, dict[str, Any]]] = []

    def _call(self, name: str, arg: Any, **kwargs: Any) -> Any:
        self.calls.append((name, arg, kwargs))
        return self.payloads[name]

    def scrape(self, url: str, **kw: Any) -> Any:
        return self._call("scrape", url, **kw)

    def map(self, url: str, **kw: Any) -> Any:
        return self._call("map", url, **kw)

    def crawl(self, url: str, **kw: Any) -> Any:
        return self._call("crawl", url, **kw)


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch):
    def _apply(**payloads: Any) -> FakeClient:
        fake = FakeClient(**payloads)
        monkeypatch.setattr("llama_index.readers.hydrafetch.base.Hydrafetch", lambda *a, **k: fake)
        return fake

    return _apply


def test_is_a_llamaindex_reader() -> None:
    assert issubclass(HydrafetchReader, BaseReader)
    assert HydrafetchReader.class_name() == "HydrafetchReader"
    assert HydrafetchReader(api_key="k").is_remote is True


def test_scrape_maps_text_and_metadata(patched) -> None:
    patched(scrape=SCRAPE)
    docs = HydrafetchReader(api_key="k").load_data("https://example.com")

    assert len(docs) == 1
    assert docs[0].text == "# Example\n\nBody text."
    assert docs[0].metadata["source"] == "https://example.com"
    assert docs[0].metadata["site_name"] == "Example"
    assert docs[0].metadata["word_count"] == 3
    assert docs[0].metadata["status"] == 200


def test_accepts_many_urls(patched) -> None:
    fake = patched(scrape=SCRAPE)
    docs = HydrafetchReader(api_key="k").load_data(
        ["https://example.com/1", "https://example.com/2"]
    )

    assert len(docs) == 2
    assert [c[1] for c in fake.calls] == ["https://example.com/1", "https://example.com/2"]


def test_requests_the_chosen_format(patched) -> None:
    fake = patched(scrape=SCRAPE)
    HydrafetchReader(api_key="k", content_format="html").load_data("https://example.com")

    assert fake.calls[0][2]["formats"] == ["html"]


def test_map_mode_yields_one_document_per_link(patched) -> None:
    patched(map=MAP)
    docs = HydrafetchReader(api_key="k").load_data("https://example.com", mode="map")

    assert [d.text for d in docs] == ["https://example.com/a", "https://example.com/b"]
    assert docs[0].metadata["lastmod"] == "2026-08-01"


def test_crawl_skips_error_and_empty_pages(patched) -> None:
    patched(crawl=CRAWL)
    docs = HydrafetchReader(api_key="k").load_data("https://example.com", mode="crawl")

    assert [d.text for d in docs] == ["A"]
    assert docs[0].metadata["depth"] == 1


def test_crawl_keeps_error_pages_when_asked(patched) -> None:
    patched(crawl=CRAWL)
    docs = HydrafetchReader(api_key="k", raise_for_status=False).load_data(
        "https://example.com", mode="crawl"
    )

    assert [d.text for d in docs] == ["A", "B"]


def test_refuses_an_error_page_by_default(patched) -> None:
    patched(scrape=NOT_FOUND)
    with pytest.raises(ValueError, match="HTTP 404"):
        HydrafetchReader(api_key="k").load_data("https://example.com/gone")


def test_loads_an_error_page_when_asked(patched) -> None:
    patched(scrape=NOT_FOUND)
    docs = HydrafetchReader(api_key="k", raise_for_status=False).load_data(
        "https://example.com/gone"
    )

    assert docs[0].metadata["status"] == 404


def test_rejects_unknown_mode(patched) -> None:
    patched(scrape=SCRAPE)
    with pytest.raises(ValueError, match="mode must be"):
        HydrafetchReader(api_key="k").load_data("https://example.com", mode="nope")


def test_params_merge_constructor_and_call(patched) -> None:
    fake = patched(scrape=SCRAPE)
    reader = HydrafetchReader(api_key="k", params={"onlyMainContent": True})
    reader.load_data("https://example.com", blockAds=True)

    assert fake.calls[0][2]["onlyMainContent"] is True
    assert fake.calls[0][2]["blockAds"] is True


def test_lazy_load_streams(patched) -> None:
    patched(map=MAP)
    stream = HydrafetchReader(api_key="k").lazy_load_data("https://example.com", mode="map")

    assert next(stream).text == "https://example.com/a"
