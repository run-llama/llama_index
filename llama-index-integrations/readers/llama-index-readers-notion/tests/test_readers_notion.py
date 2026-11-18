from unittest.mock import MagicMock

from llama_index.core.readers.base import BaseReader
from llama_index.readers.notion import NotionPageReader
from llama_index.readers.notion.base import DEFAULT_BASE_URL


def test_class():
    names_of_base_classes = [b.__name__ for b in NotionPageReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def _search_response() -> MagicMock:
    res = MagicMock()
    res.json.return_value = {"results": [{"id": "page-1"}], "next_cursor": None}
    res.raise_for_status.return_value = None
    res.status_code = 200
    return res


def test_default_base_url():
    reader = NotionPageReader(integration_token="secret")
    assert reader.base_url == DEFAULT_BASE_URL


def test_base_url_from_env(monkeypatch):
    monkeypatch.setenv("NOTION_BASE_URL", "http://localhost:9000")
    reader = NotionPageReader(integration_token="secret")
    assert reader.base_url == "http://localhost:9000"


def test_explicit_base_url_overrides_env(monkeypatch):
    monkeypatch.setenv("NOTION_BASE_URL", "http://from-env:9000")
    reader = NotionPageReader(
        integration_token="secret", base_url="http://explicit:8000"
    )
    assert reader.base_url == "http://explicit:8000"


def test_base_url_trailing_slash_stripped():
    reader = NotionPageReader(integration_token="secret", base_url="http://host:8000/")
    assert reader.base_url == "http://host:8000"


def test_custom_base_url_used_for_requests(mocker):
    reader = NotionPageReader(
        integration_token="secret", base_url="http://localhost:8000/notion"
    )
    request = mocker.patch(
        "llama_index.readers.notion.base.requests.request",
        return_value=_search_response(),
    )

    reader.search("hello")

    # requests.request(method, url, headers=..., json=...) — url is the 2nd positional arg
    called_url = request.call_args.args[1]
    assert called_url == "http://localhost:8000/notion/v1/search"


def test_default_base_url_used_for_requests(mocker):
    reader = NotionPageReader(integration_token="secret")
    request = mocker.patch(
        "llama_index.readers.notion.base.requests.request",
        return_value=_search_response(),
    )

    reader.search("hello")

    assert request.call_args.args[1] == "https://api.notion.com/v1/search"
