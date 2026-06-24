from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.searchapi import SearchApiToolSpec


def test_class():
    names_of_base_classes = [b.__name__ for b in SearchApiToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_default_engine():
    tool = SearchApiToolSpec(api_key="fake-key")
    assert tool.engine == "google"


def test_custom_engine():
    tool = SearchApiToolSpec(api_key="fake-key", engine="google_news")
    assert tool.engine == "google_news"
