import sys
import types
import pytest  # type: ignore

from llama_index.readers.web.firecrawl_web.base import FireCrawlWebReader


def _install_fake_firecrawl(FirecrawlClass) -> None:
    mod = types.ModuleType("firecrawl")
    mod.Firecrawl = FirecrawlClass
    mod.__version__ = "test"
    sys.modules["firecrawl"] = mod


class _Link:
    def __init__(self, url: str, title: str = "", description: str = "") -> None:
        self.url = url
        self.title = title
        self.description = description


class _MapResponse:
    def __init__(self, links):
        self.links = links


def test_class_name_returns_expected():
    class Firecrawl:
        def __init__(self, *args, **kwargs) -> None:
            pass

    _install_fake_firecrawl(Firecrawl)
    assert FireCrawlWebReader.class_name() == "Firecrawl_reader"


def test_init_uses_api_key_and_url():
    class Firecrawl:
        def __init__(self, api_key: str, api_url: str = None) -> None:  # type: ignore[assignment]
            self.api_key = api_key
            self.api_url = api_url

    _install_fake_firecrawl(Firecrawl)
    reader = FireCrawlWebReader(
        api_key="KEY123", api_url="https://api.example", mode="scrape"
    )
    assert reader.firecrawl.api_key == "KEY123"
    assert reader.firecrawl.api_url == "https://api.example"


def test_scrape_mode_with_dict_response_includes_text_and_metadata():
    scrape_called = {}

    class Firecrawl:
        def __init__(self, *_, **__):
            pass

        def scrape(self, url: str, **kwargs):
            scrape_called["url"] = url
            scrape_called["kwargs"] = kwargs
            return {
                "success": True,
                "warning": None,
                "data": {
                    "markdown": "Hello MD",
                    "metadata": {"a": 1},
                    "links": ["x"],
                },
            }

    _install_fake_firecrawl(Firecrawl)
    reader = FireCrawlWebReader(
        api_key="k", mode="scrape", params={"formats": ["markdown"]}
    )
    docs = reader.load_data(url="https://site")
    assert len(docs) == 1
    assert docs[0].text == "Hello MD"
    assert docs[0].metadata.get("a") == 1
    assert docs[0].metadata.get("success") is True
    assert scrape_called["url"] == "https://site"
    # Allow additional kwargs (e.g., integration flag) but ensure formats are passed through
    assert scrape_called["kwargs"].get("formats") == ["markdown"]
    assert scrape_called["kwargs"].get("integration") == "llamaindex"


def test_scrape_mode_with_object_response_includes_text_and_metadata():
    class Meta:
        def __init__(self):
            self.lang = "en"

        def model_dump(self):
            return {"lang": self.lang}

    class ScrapeObj:
        def __init__(self):
            self.markdown = "Obj MD"
            self.metadata = Meta()
            self.links = ["y"]
            self.warning = None

    class Firecrawl:
        def __init__(self, *_, **__):
            pass

        def scrape(self, url: str, **kwargs):
            return ScrapeObj()

    _install_fake_firecrawl(Firecrawl)
    reader = FireCrawlWebReader(api_key="k", mode="scrape")
    docs = reader.load_data(url="https://site")
    assert len(docs) == 1
    assert docs[0].text == "Obj MD"
    assert docs[0].metadata.get("lang") == "en"


def test_crawl_mode_strips_maxDepth_and_maps_docs():
    last_kwargs = {}

    class Firecrawl:
        def __init__(self, *_, **__):
            pass

        def crawl(self, url: str, **kwargs):
            last_kwargs.update(kwargs)
            return {
                "data": [
                    {"markdown": "A", "metadata": {"u": url}},
                    {"content": "B", "metadata": {"n": 2}},
                ]
            }

    _install_fake_firecrawl(Firecrawl)
    reader = FireCrawlWebReader(
        api_key="k", mode="crawl", params={"maxDepth": 2, "limit": 1}
    )
    docs = reader.load_data(url="https://site/x")
    assert [d.text for d in docs] == ["A", "B"]
    assert "maxDepth" not in last_kwargs
    assert last_kwargs.get("limit") == 1


def test_map_mode_success_yields_link_documents():
    class Firecrawl:
        def __init__(self, *_, **__):
            pass

        def map(self, url: str, **kwargs):  # type: ignore[override]
            return _MapResponse(
                [
                    _Link(url="https://a", title="T1", description="D1"),
                    _Link(url="https://b", title="", description="D2"),
                ]
            )

    _install_fake_firecrawl(Firecrawl)
    reader = FireCrawlWebReader(api_key="k", mode="map", params={"limit": 2})
    docs = reader.load_data(url="https://root")
    assert len(docs) == 2
    assert docs[0].metadata["source"] == "map"
    assert docs[0].metadata["url"] == "https://a"
    # text falls back to title/description/url
    assert docs[1].text == "D2"


def test_map_mode_error_returns_single_error_document():
    class Firecrawl:
        def __init__(self, *_, **__):
            pass

        def map(self, url: str, **kwargs):  # type: ignore[override]
            return {"success": False, "error": "rate limit"}

    _install_fake_firecrawl(Firecrawl)
    reader = FireCrawlWebReader(api_key="k", mode="map")
    docs = reader.load_data(url="https://root")
    assert len(docs) == 1
    assert "rate limit" in docs[0].text
    assert docs[0].metadata["error"] == "rate limit"


def test_search_mode_with_dict_success_and_markdown_fallbacks():
    passed_kwargs = {}

    class Firecrawl:
        def __init__(self, *_, **__):
            pass

        def search(self, query: str, **kwargs):
            passed_kwargs.update(kwargs)
            return {
                "success": True,
                "data": [
                    {"title": "A", "url": "u1", "markdown": "M1", "metadata": {"x": 1}},
                    {"title": "B", "url": "u2", "description": "D2"},
                ],
            }

    _install_fake_firecrawl(Firecrawl)
    reader = FireCrawlWebReader(
        api_key="k", mode="search", params={"query": "dup", "region": "us"}
    )
    docs = reader.load_data(query="q")
    assert [d.text for d in docs] == ["M1", "D2"]
    # ensure reader removed duplicate 'query' from params before call
    assert "query" not in passed_kwargs
    assert passed_kwargs.get("region") == "us"


def test_search_mode_with_dict_failure_returns_error_document():
    class Firecrawl:
        def __init__(self, *_, **__):
            pass

        def search(self, query: str, **kwargs):
            return {"success": False, "warning": "bad query"}

    _install_fake_firecrawl(Firecrawl)
    reader = FireCrawlWebReader(api_key="k", mode="search")
    docs = reader.load_data(query="q")
    assert len(docs) == 1
    assert "unsuccessful" in docs[0].text
    assert docs[0].metadata["error"] == "bad query"


def test_search_mode_with_sdk_object_lists():
    class Item:
        def __init__(self, url: str, title: str, description: str) -> None:
            self.url = url
            self.title = title
            self.description = description
            self.rank = 7

    class SearchResp:
        def __init__(self):
            self.web = [Item("https://a", "T1", "D1")]
            self.news = []
            self.images = [Item("https://img", "", "image desc")]

    class Firecrawl:
        def __init__(self, *_, **__):
            pass

        def search(self, query: str, **kwargs):
            return SearchResp()

    _install_fake_firecrawl(Firecrawl)
    reader = FireCrawlWebReader(api_key="k", mode="search")
    docs = reader.load_data(query="q")
    assert len(docs) == 2
    types = {d.metadata.get("search_type") for d in docs}
    assert types == {"web", "images"}
    assert any(d.metadata.get("rank") == 7 for d in docs)


def test_extract_mode_success_with_sources_and_status():
    class Firecrawl:
        def __init__(self, *_, **__):
            pass

        def extract(self, *, urls, **payload):
            # Accept additional fields such as integration while verifying prompt
            assert payload.get("prompt") == "Do it"
            assert urls == ["https://a", "https://b"]
            return {
                "success": True,
                "status": "ok",
                "expiresAt": "2030-01-01",
                "data": {"k1": "v1", "k2": 2},
                "sources": {"https://a": {"score": 1.0}},
            }

    _install_fake_firecrawl(Firecrawl)
    reader = FireCrawlWebReader(api_key="k", mode="extract", params={"prompt": "Do it"})
    docs = reader.load_data(urls=["https://a", "https://b"])
    assert len(docs) == 1
    assert "k1: v1" in docs[0].text
    assert docs[0].metadata["status"] == "ok"
    assert "sources" in docs[0].metadata


def test_extract_mode_success_no_data_yields_notice():
    class Firecrawl:
        def __init__(self, *_, **__):
            pass

        def extract(self, *, urls, **payload):
            return {"success": True, "data": {}}

    _install_fake_firecrawl(Firecrawl)
    reader = FireCrawlWebReader(api_key="k", mode="extract", params={"prompt": "x"})
    docs = reader.load_data(urls=["https://x"])
    assert len(docs) == 1
    assert "no data" in docs[0].text.lower()


def test_extract_mode_failure_returns_error_document():
    class Firecrawl:
        def __init__(self, *_, **__):
            pass

        def extract(self, *, urls, **payload):
            return {"success": False, "warning": "no quota"}

    _install_fake_firecrawl(Firecrawl)
    reader = FireCrawlWebReader(api_key="k", mode="extract", params={"prompt": "x"})
    docs = reader.load_data(urls=["https://x"])
    assert len(docs) == 1
    assert docs[0].metadata["error"] == "no quota"


def test_invalid_mode_raises_value_error():
    class Firecrawl:
        def __init__(self, *_, **__):
            pass

    _install_fake_firecrawl(Firecrawl)
    reader = FireCrawlWebReader(api_key="k", mode="invalid")
    with pytest.raises(ValueError):
        reader.load_data(url="https://x")


def test_argument_validation_requires_exactly_one_of_url_query_urls():
    class Firecrawl:
        def __init__(self, *_, **__):
            pass

    _install_fake_firecrawl(Firecrawl)
    reader = FireCrawlWebReader(api_key="k", mode="scrape")
    with pytest.raises(ValueError):
        reader.load_data()  # none
    with pytest.raises(ValueError):
        reader.load_data(url="u", query="q")  # two
    with pytest.raises(ValueError):
        reader.load_data(url="u", urls=["u"])  # two
