
import pytest
from unittest.mock import MagicMock, patch
import sys
from typing import Generator

@pytest.fixture
def mock_tzafon() -> Generator[MagicMock, None, None]:
    with patch.dict(sys.modules, {"tzafon": MagicMock()}):
        yield sys.modules["tzafon"]

def test_tzafon_web_reader_init(mock_tzafon: MagicMock) -> None:
    from llama_index.readers.web import TzafonWebReader
    
    # Setup mock
    mock_computer_cls = MagicMock()
    mock_tzafon.Computer = mock_computer_cls
    
    reader = TzafonWebReader(api_key="test_key")
    assert reader.api_key == "test_key"
    mock_computer_cls.assert_called_with(api_key="test_key")

def test_tzafon_web_reader_load_data(mock_tzafon: MagicMock) -> None:
    from llama_index.readers.web import TzafonWebReader
    
    # Setup Tzafon mock
    mock_computer_cls = MagicMock()
    mock_tzafon.Computer = mock_computer_cls
    mock_computer_instance = mock_computer_cls.return_value
    mock_computer_instance.create.return_value.id = "test_id"

    
    with patch("playwright.sync_api.sync_playwright") as mock_sync_playwright:
        mock_playwright_context = mock_sync_playwright.return_value.__enter__.return_value
        mock_browser = mock_playwright_context.chromium.connect_over_cdp.return_value
        mock_context = MagicMock()
        mock_browser.contexts = [mock_context]
        
        mock_page = MagicMock()
        mock_context.new_page.return_value = mock_page
        mock_page.inner_text.return_value = "Page Content"
        mock_page.content.return_value = "<html>Page Content</html>"
        
        reader = TzafonWebReader(api_key="test_key")
        docs = reader.load_data(urls=["https://example.com"])
        
        assert len(docs) == 1
        assert docs[0].text == "Page Content"
        assert docs[0].metadata["url"] == "https://example.com"
