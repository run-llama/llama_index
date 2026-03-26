from unittest.mock import MagicMock

from llama_index.tools.aws_bedrock_agentcore.browser.utils import get_current_page


class TestGetCurrentPage:
    def test_no_contexts(self):
        mock_browser = MagicMock()
        mock_browser.contexts = []
        mock_context = MagicMock()
        mock_page = MagicMock()
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page

        result = get_current_page(mock_browser)

        assert result == mock_page
        mock_browser.new_context.assert_called_once()
        mock_context.new_page.assert_called_once()

    def test_context_no_pages(self):
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_context.pages = []
        mock_browser.contexts = [mock_context]
        mock_page = MagicMock()
        mock_context.new_page.return_value = mock_page

        result = get_current_page(mock_browser)

        assert result == mock_page
        mock_browser.new_context.assert_not_called()
        mock_context.new_page.assert_called_once()

    def test_context_with_pages(self):
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page1 = MagicMock()
        mock_page2 = MagicMock()
        mock_context.pages = [mock_page1, mock_page2]
        mock_browser.contexts = [mock_context]

        result = get_current_page(mock_browser)

        assert result == mock_page2  # Should return the last page
        mock_browser.new_context.assert_not_called()
        mock_context.new_page.assert_not_called()
