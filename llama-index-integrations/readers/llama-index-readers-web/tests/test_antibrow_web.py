"""Tests for AntibrowWebReader."""

from unittest.mock import MagicMock, patch

import pytest

from llama_index.core.schema import Document
from llama_index.readers.web import AntibrowWebReader


@pytest.fixture
def mock_launch():
    """Patch the SDK entry point so no browser is started."""
    page = MagicMock()
    page.url = "https://example.com/"
    page.title.return_value = "Example Domain"
    page.goto.return_value = MagicMock(status=200)
    page.locator.return_value.first.inner_text.return_value = "Example Domain"
    browser = MagicMock()
    browser.new_page.return_value = page
    with patch("antibrow.launch", return_value=browser) as launch:
        yield {"launch": launch, "browser": browser, "page": page}


def test_reader_returns_documents_with_metadata(mock_launch):
    reader = AntibrowWebReader(api_key="test_key", profile="test-profile")
    docs = reader.load_data(urls=["https://example.com"])

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].text == "Example Domain"
    assert docs[0].metadata == {
        "url": "https://example.com/",
        "title": "Example Domain",
        "status": 200,
        "profile": "test-profile",
    }


def test_the_profile_is_launched_with_its_options(mock_launch):
    reader = AntibrowWebReader(
        api_key="test_key", profile="p1", proxy="http://user:pass@host:5001", temporary=True
    )
    reader.load_data(urls=["https://example.com"])

    args, kwargs = mock_launch["launch"].call_args
    assert args[0] == "p1"
    assert kwargs["proxy"] == "http://user:pass@host:5001"
    assert kwargs["temporary"] is True
    assert kwargs["focus_window"] is False


def test_one_browser_serves_every_url(mock_launch):
    reader = AntibrowWebReader(api_key="k")
    docs = reader.load_data(urls=["https://example.com", "https://example.org"])

    assert len(docs) == 2
    assert mock_launch["launch"].call_count == 1
    assert mock_launch["page"].goto.call_count == 2


def test_selector_is_used_when_given(mock_launch):
    reader = AntibrowWebReader(api_key="k")
    reader.load_data(urls=["https://example.com"], selector="main")
    mock_launch["page"].locator.assert_called_with("main")


def test_the_browser_is_closed_even_when_a_page_fails(mock_launch):
    mock_launch["page"].goto.side_effect = RuntimeError("navigation failed")
    reader = AntibrowWebReader(api_key="k")

    with pytest.raises(RuntimeError):
        reader.load_data(urls=["https://example.com"])
    mock_launch["browser"].close.assert_called_once()
