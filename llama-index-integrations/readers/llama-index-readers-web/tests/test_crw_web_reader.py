import sys
import types
import pytest  # type: ignore

from llama_index.readers.web.crw_web.base import CrwWebReader


def _install_fake_crw(CrwClientClass) -> None:
    mod = types.ModuleType("crw")
    mod.CrwClient = CrwClientClass
    mod.__version__ = "test"
    sys.modules["crw"] = mod


def test_class_name_returns_expected():
    class CrwClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

    _install_fake_crw(CrwClient)
    assert CrwWebReader.class_name() == "Crw_reader"


def test_init_uses_api_key_and_url():
    class CrwClient:
        def __init__(self, api_key: str = None, api_url: str = None) -> None:  # type: ignore[assignment]
            self.api_key = api_key
            self.api_url = api_url

    _install_fake_crw(CrwClient)
    reader = CrwWebReader(
        api_key="KEY123", api_url="https://api.example", mode="scrape"
    )
    assert reader.crw.api_key == "KEY123"
    assert reader.crw.api_url == "https://api.example"


def test_init_without_api_key_relies_on_env():
    class CrwClient:
        def __init__(self, api_key: str = None, api_url: str = None) -> None:  # type: ignore[assignment]
            self.api_key = api_key
            self.api_url = api_url

    _install_fake_crw(CrwClient)
    reader = CrwWebReader(mode="scrape")
    # api_key not forwarded when omitted -> client reads CRW_API_KEY from env
    assert reader.crw.api_key is None


def test_scrape_mode_with_dict_response_includes_text_and_metadata():
    scrape_called = {}

    class CrwClient:
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

    _install_fake_crw(CrwClient)
    reader = CrwWebReader(
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

    class CrwClient:
        def __init__(self, *_, **__):
            pass

        def scrape(self, url: str, **kwargs):
            return ScrapeObj()

    _install_fake_crw(CrwClient)
    reader = CrwWebReader(api_key="k", mode="scrape")
    docs = reader.load_data(url="https://site")
    assert len(docs) == 1
    assert docs[0].text == "Obj MD"
    assert docs[0].metadata.get("lang") == "en"


def test_crawl_mode_with_list_response_maps_docs():
    last_kwargs = {}

    class CrwClient:
        def __init__(self, *_, **__):
            pass

        def crawl(self, url: str, **kwargs):
            last_kwargs.update(kwargs)
            return [
                {"markdown": "A", "metadata": {"u": url}},
                {"content": "B", "metadata": {"n": 2}},
            ]

    _install_fake_crw(CrwClient)
    reader = CrwWebReader(
        api_key="k", mode="crawl", params={"maxPages": 1}
    )
    docs = reader.load_data(url="https://site/x")
    assert [d.text for d in docs] == ["A", "B"]
    assert last_kwargs.get("maxPages") == 1


def test_crawl_mode_with_dict_envelope_maps_docs():
    class CrwClient:
        def __init__(self, *_, **__):
            pass

        def crawl(self, url: str, **kwargs):
            return {
                "data": [
                    {"markdown": "A", "metadata": {"u": url}},
                    {"content": "B", "metadata": {"n": 2}},
                ]
            }

    _install_fake_crw(CrwClient)
    reader = CrwWebReader(api_key="k", mode="crawl")
    docs = reader.load_data(url="https://site/x")
    assert [d.text for d in docs] == ["A", "B"]


def test_map_mode_with_list_of_strings_yields_link_documents():
    class CrwClient:
        def __init__(self, *_, **__):
            pass

        def map(self, url: str, **kwargs):  # type: ignore[override]
            return ["https://a", "https://b"]

    _install_fake_crw(CrwClient)
    reader = CrwWebReader(api_key="k", mode="map", params={"limit": 2})
    docs = reader.load_data(url="https://root")
    assert len(docs) == 2
    assert docs[0].metadata["source"] == "map"
    assert docs[0].metadata["url"] == "https://a"
    # text falls back to url when no title/description
    assert docs[1].text == "https://b"


def test_map_mode_error_returns_single_error_document():
    class CrwClient:
        def __init__(self, *_, **__):
            pass

        def map(self, url: str, **kwargs):  # type: ignore[override]
            return {"success": False, "error": "rate limit"}

    _install_fake_crw(CrwClient)
    reader = CrwWebReader(api_key="k", mode="map")
    docs = reader.load_data(url="https://root")
    assert len(docs) == 1
    assert "rate limit" in docs[0].text
    assert docs[0].metadata["error"] == "rate limit"


def test_search_mode_with_list_and_markdown_fallbacks():
    passed_kwargs = {}

    class CrwClient:
        def __init__(self, *_, **__):
            pass

        def search(self, query: str, **kwargs):
            passed_kwargs.update(kwargs)
            return [
                {"title": "A", "url": "u1", "markdown": "M1", "metadata": {"x": 1}},
                {"title": "B", "url": "u2", "description": "D2"},
            ]

    _install_fake_crw(CrwClient)
    reader = CrwWebReader(
        api_key="k", mode="search", params={"query": "dup", "region": "us"}
    )
    docs = reader.load_data(query="q")
    assert [d.text for d in docs] == ["M1", "D2"]
    # ensure reader removed duplicate 'query' from params before call
    assert "query" not in passed_kwargs
    assert passed_kwargs.get("region") == "us"


def test_search_mode_with_dict_envelope_failure_returns_error_document():
    class CrwClient:
        def __init__(self, *_, **__):
            pass

        def search(self, query: str, **kwargs):
            return {"success": False, "warning": "bad query"}

    _install_fake_crw(CrwClient)
    reader = CrwWebReader(api_key="k", mode="search")
    docs = reader.load_data(query="q")
    assert len(docs) == 1
    assert "unsuccessful" in docs[0].text
    assert docs[0].metadata["error"] == "bad query"


def test_invalid_mode_raises_value_error():
    class CrwClient:
        def __init__(self, *_, **__):
            pass

    _install_fake_crw(CrwClient)
    reader = CrwWebReader(api_key="k", mode="invalid")
    with pytest.raises(ValueError):
        reader.load_data(url="https://x")


def test_argument_validation_requires_exactly_one_of_url_query():
    class CrwClient:
        def __init__(self, *_, **__):
            pass

    _install_fake_crw(CrwClient)
    reader = CrwWebReader(api_key="k", mode="scrape")
    with pytest.raises(ValueError):
        reader.load_data()  # none
    with pytest.raises(ValueError):
        reader.load_data(url="u", query="q")  # two
