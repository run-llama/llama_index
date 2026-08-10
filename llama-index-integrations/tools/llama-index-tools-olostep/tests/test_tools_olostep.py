from unittest.mock import MagicMock, patch

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.olostep import OlostepToolSpec


def test_class():
    names_of_base_classes = [b.__name__ for b in OlostepToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_spec_functions():
    assert OlostepToolSpec.spec_functions == ["scrape", "answer", "crawl", "map_site"]


@patch("olostep.Olostep")
def test_scrape(mock_olostep):
    mock_client = MagicMock()
    mock_result = MagicMock(spec=["markdown_content"])
    mock_result.markdown_content = "# Example Domain"
    mock_client.scrapes.create.return_value = mock_result
    mock_olostep.return_value = mock_client

    tool = OlostepToolSpec(api_key="test-key")
    docs = tool.scrape("https://example.com")

    assert len(docs) == 1
    assert docs[0].text == "# Example Domain"
    assert docs[0].metadata["url"] == "https://example.com"


@patch("olostep.Olostep")
def test_answer(mock_olostep):
    mock_client = MagicMock()
    mock_result = MagicMock(spec=["json_content"])
    mock_result.json_content = '{"result":"Python 3.14"}'
    mock_client.answers.create.return_value = mock_result
    mock_olostep.return_value = mock_client

    tool = OlostepToolSpec(api_key="test-key")
    docs = tool.answer("What is the latest version of Python?")

    assert len(docs) == 1
    assert "Python 3.14" in docs[0].text


@patch("olostep.Olostep")
def test_map_site(mock_olostep):
    mock_client = MagicMock()
    mock_sitemap = MagicMock()
    mock_sitemap.urls = lambda: ["https://a.com", "https://b.com"]
    mock_client.maps.create.return_value = mock_sitemap
    mock_olostep.return_value = mock_client

    tool = OlostepToolSpec(api_key="test-key")
    docs = tool.map_site("https://a.com")

    assert len(docs) == 1
    assert "https://a.com" in docs[0].text
    assert "https://b.com" in docs[0].text