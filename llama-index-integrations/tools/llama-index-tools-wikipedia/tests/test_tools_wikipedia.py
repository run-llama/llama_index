from unittest.mock import Mock, patch
from urllib.parse import urlsplit

import pytest
import wikipedia

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.wikipedia import WikipediaToolSpec


def test_class():
    names_of_base_classes = [b.__name__ for b in WikipediaToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


@pytest.mark.parametrize(("previous_lang", "lang"), [("en", "de"), ("de", "en")])
def test_search_uses_requested_language(previous_lang: str, lang: str) -> None:
    wikipedia.set_lang(previous_lang)
    tool = WikipediaToolSpec()
    endpoints = []

    def search(query: str) -> list[str]:
        endpoints.append(urlsplit(wikipedia.wikipedia.API_URL).hostname)
        return ["Berlin"]

    try:
        with (
            patch.object(wikipedia, "search", side_effect=search),
            patch.object(wikipedia, "page", return_value=Mock(content="Berlin page")),
        ):
            assert tool.search_data("Berlin", lang=lang) == "Berlin page"
        assert endpoints == [f"{lang}.wikipedia.org"]
    finally:
        wikipedia.set_lang("en")
