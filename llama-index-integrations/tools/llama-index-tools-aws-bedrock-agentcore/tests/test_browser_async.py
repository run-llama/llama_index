import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from llama_index.tools.aws_bedrock_agentcore import AgentCoreBrowserToolSpec


class TestAsyncBrowserFunctions:
    @pytest.mark.asyncio
    async def test_anavigate_browser_invalid_url(self):
        """Test anavigate_browser with an invalid URL scheme."""
        tool_spec = AgentCoreBrowserToolSpec()

        result = await tool_spec.anavigate_browser(
            url="ftp://example.com", thread_id="test-thread"
        )

        assert "URL scheme must be 'http' or 'https'" in result

    @pytest.mark.asyncio
    @patch("llama_index.tools.aws_bedrock_agentcore.browser.base.aget_current_page")
    async def test_anavigate_browser(self, mock_aget_current_page):
        """Test anavigate_browser with a valid URL."""
        mock_session_manager = MagicMock()
        mock_browser = AsyncMock()
        mock_page = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200

        mock_session_manager.get_async_browser = AsyncMock(return_value=mock_browser)
        mock_session_manager.release_async_browser = AsyncMock()
        mock_aget_current_page.return_value = mock_page
        mock_page.goto = AsyncMock(return_value=mock_response)

        tool_spec = AgentCoreBrowserToolSpec()
        tool_spec._session_manager = mock_session_manager

        result = await tool_spec.anavigate_browser(
            url="https://example.com", thread_id="test-thread"
        )

        mock_session_manager.get_async_browser.assert_awaited_once_with("test-thread")
        mock_aget_current_page.assert_awaited_once_with(mock_browser)
        mock_page.goto.assert_awaited_once_with("https://example.com")
        mock_session_manager.release_async_browser.assert_awaited_once_with(
            "test-thread"
        )
        assert "Navigated to https://example.com with status code 200" in result

    @pytest.mark.asyncio
    async def test_anavigate_browser_exception(self):
        """Test anavigate_browser with an exception."""
        mock_session_manager = MagicMock()
        mock_session_manager.get_async_browser = AsyncMock(
            side_effect=Exception("Test error")
        )

        tool_spec = AgentCoreBrowserToolSpec()
        tool_spec._session_manager = mock_session_manager

        result = await tool_spec.anavigate_browser(
            url="https://example.com", thread_id="test-thread"
        )

        assert "Error navigating to URL: Test error" in result

    @pytest.mark.asyncio
    @patch("llama_index.tools.aws_bedrock_agentcore.browser.base.aget_current_page")
    async def test_aclick_element(self, mock_aget_current_page):
        """Test aclick_element with a valid selector."""
        mock_session_manager = MagicMock()
        mock_browser = AsyncMock()
        mock_page = AsyncMock()

        mock_session_manager.get_async_browser = AsyncMock(return_value=mock_browser)
        mock_session_manager.release_async_browser = AsyncMock()
        mock_aget_current_page.return_value = mock_page
        mock_page.click = AsyncMock()

        tool_spec = AgentCoreBrowserToolSpec()
        tool_spec._session_manager = mock_session_manager

        result = await tool_spec.aclick_element(
            selector="#button", thread_id="test-thread"
        )

        mock_session_manager.get_async_browser.assert_awaited_once_with("test-thread")
        mock_aget_current_page.assert_awaited_once_with(mock_browser)
        mock_page.click.assert_awaited_once_with("#button", timeout=5000)
        mock_session_manager.release_async_browser.assert_awaited_once_with(
            "test-thread"
        )
        assert "Clicked on element with selector '#button'" in result

    @pytest.mark.asyncio
    @patch("llama_index.tools.aws_bedrock_agentcore.browser.base.aget_current_page")
    async def test_aclick_element_not_found(self, mock_aget_current_page):
        """Test aclick_element with a selector that doesn't match any elements."""
        mock_session_manager = MagicMock()
        mock_browser = AsyncMock()
        mock_page = AsyncMock()

        mock_session_manager.get_async_browser = AsyncMock(return_value=mock_browser)
        mock_session_manager.release_async_browser = AsyncMock()
        mock_aget_current_page.return_value = mock_page
        mock_page.click = AsyncMock(side_effect=Exception("Element not found"))

        tool_spec = AgentCoreBrowserToolSpec()
        tool_spec._session_manager = mock_session_manager

        result = await tool_spec.aclick_element(
            selector="#button", thread_id="test-thread"
        )

        mock_session_manager.get_async_browser.assert_awaited_once_with("test-thread")
        mock_aget_current_page.assert_awaited_once_with(mock_browser)
        mock_page.click.assert_awaited_once_with("#button", timeout=5000)
        mock_session_manager.release_async_browser.assert_awaited_once_with(
            "test-thread"
        )
        assert "Unable to click on element with selector '#button'" in result

    @pytest.mark.asyncio
    @patch("llama_index.tools.aws_bedrock_agentcore.browser.base.aget_current_page")
    async def test_aextract_text_whole_page(self, mock_aget_current_page):
        """Test aextract_text for the whole page."""
        mock_session_manager = MagicMock()
        mock_browser = AsyncMock()
        mock_page = AsyncMock()

        mock_session_manager.get_async_browser = AsyncMock(return_value=mock_browser)
        mock_session_manager.release_async_browser = AsyncMock()
        mock_aget_current_page.return_value = mock_page
        mock_page.content = AsyncMock(
            return_value="<html><body>Hello World</body></html>"
        )

        tool_spec = AgentCoreBrowserToolSpec()
        tool_spec._session_manager = mock_session_manager

        result = await tool_spec.aextract_text(thread_id="test-thread")

        mock_session_manager.get_async_browser.assert_awaited_once_with("test-thread")
        mock_aget_current_page.assert_awaited_once_with(mock_browser)
        mock_page.content.assert_awaited_once()
        mock_session_manager.release_async_browser.assert_awaited_once_with(
            "test-thread"
        )
        assert result == "<html><body>Hello World</body></html>"

    @pytest.mark.asyncio
    @patch("llama_index.tools.aws_bedrock_agentcore.browser.base.aget_current_page")
    async def test_aextract_text_with_selector(self, mock_aget_current_page):
        """Test aextract_text with a selector."""
        mock_session_manager = MagicMock()
        mock_browser = AsyncMock()
        mock_page = AsyncMock()
        mock_element = AsyncMock()

        mock_session_manager.get_async_browser = AsyncMock(return_value=mock_browser)
        mock_session_manager.release_async_browser = AsyncMock()
        mock_aget_current_page.return_value = mock_page
        mock_page.query_selector = AsyncMock(return_value=mock_element)
        mock_element.text_content = AsyncMock(return_value="Hello World")

        tool_spec = AgentCoreBrowserToolSpec()
        tool_spec._session_manager = mock_session_manager

        result = await tool_spec.aextract_text(
            selector="#content", thread_id="test-thread"
        )

        mock_session_manager.get_async_browser.assert_awaited_once_with("test-thread")
        mock_aget_current_page.assert_awaited_once_with(mock_browser)
        mock_page.query_selector.assert_awaited_once_with("#content")
        mock_element.text_content.assert_awaited_once()
        mock_session_manager.release_async_browser.assert_awaited_once_with(
            "test-thread"
        )
        assert result == "Hello World"

    @pytest.mark.asyncio
    @patch("llama_index.tools.aws_bedrock_agentcore.browser.base.aget_current_page")
    async def test_aextract_text_selector_not_found(self, mock_aget_current_page):
        """Test aextract_text with a selector that doesn't match any elements."""
        mock_session_manager = MagicMock()
        mock_browser = AsyncMock()
        mock_page = AsyncMock()

        mock_session_manager.get_async_browser = AsyncMock(return_value=mock_browser)
        mock_session_manager.release_async_browser = AsyncMock()
        mock_aget_current_page.return_value = mock_page
        mock_page.query_selector = AsyncMock(return_value=None)

        tool_spec = AgentCoreBrowserToolSpec()
        tool_spec._session_manager = mock_session_manager

        result = await tool_spec.aextract_text(
            selector="#content", thread_id="test-thread"
        )

        mock_session_manager.get_async_browser.assert_awaited_once_with("test-thread")
        mock_aget_current_page.assert_awaited_once_with(mock_browser)
        mock_page.query_selector.assert_awaited_once_with("#content")
        mock_session_manager.release_async_browser.assert_awaited_once_with(
            "test-thread"
        )
        assert "No element found with selector '#content'" in result

    @pytest.mark.asyncio
    @patch("llama_index.tools.aws_bedrock_agentcore.browser.base.aget_current_page")
    async def test_aextract_hyperlinks(self, mock_aget_current_page):
        """Test aextract_hyperlinks."""
        mock_session_manager = MagicMock()
        mock_browser = AsyncMock()
        mock_page = AsyncMock()

        mock_session_manager.get_async_browser = AsyncMock(return_value=mock_browser)
        mock_session_manager.release_async_browser = AsyncMock()
        mock_aget_current_page.return_value = mock_page
        mock_page.eval_on_selector_all = AsyncMock(
            return_value=[
                {"text": "Link 1", "href": "https://example.com/1"},
                {"text": "Link 2", "href": "https://example.com/2"},
            ]
        )

        tool_spec = AgentCoreBrowserToolSpec()
        tool_spec._session_manager = mock_session_manager

        result = await tool_spec.aextract_hyperlinks(thread_id="test-thread")

        mock_session_manager.get_async_browser.assert_awaited_once_with("test-thread")
        mock_aget_current_page.assert_awaited_once_with(mock_browser)
        mock_page.eval_on_selector_all.assert_awaited_once()
        mock_session_manager.release_async_browser.assert_awaited_once_with(
            "test-thread"
        )
        assert "1. Link 1: https://example.com/1" in result
        assert "2. Link 2: https://example.com/2" in result

    @pytest.mark.asyncio
    @patch("llama_index.tools.aws_bedrock_agentcore.browser.base.aget_current_page")
    async def test_aextract_hyperlinks_no_links(self, mock_aget_current_page):
        """Test aextract_hyperlinks when no links are found."""
        mock_session_manager = MagicMock()
        mock_browser = AsyncMock()
        mock_page = AsyncMock()

        mock_session_manager.get_async_browser = AsyncMock(return_value=mock_browser)
        mock_session_manager.release_async_browser = AsyncMock()
        mock_aget_current_page.return_value = mock_page
        mock_page.eval_on_selector_all = AsyncMock(return_value=[])

        tool_spec = AgentCoreBrowserToolSpec()
        tool_spec._session_manager = mock_session_manager

        result = await tool_spec.aextract_hyperlinks(thread_id="test-thread")

        mock_session_manager.get_async_browser.assert_awaited_once_with("test-thread")
        mock_aget_current_page.assert_awaited_once_with(mock_browser)
        mock_page.eval_on_selector_all.assert_awaited_once()
        mock_session_manager.release_async_browser.assert_awaited_once_with(
            "test-thread"
        )
        assert "No hyperlinks found on the page" in result

    @pytest.mark.asyncio
    @patch("llama_index.tools.aws_bedrock_agentcore.browser.base.aget_current_page")
    async def test_aget_elements(self, mock_aget_current_page):
        """Test aget_elements."""
        mock_session_manager = MagicMock()
        mock_browser = AsyncMock()
        mock_page = AsyncMock()
        mock_element1 = AsyncMock()
        mock_element2 = AsyncMock()

        mock_session_manager.get_async_browser = AsyncMock(return_value=mock_browser)
        mock_session_manager.release_async_browser = AsyncMock()
        mock_aget_current_page.return_value = mock_page
        mock_page.query_selector_all = AsyncMock(
            return_value=[mock_element1, mock_element2]
        )

        mock_element1.evaluate = AsyncMock(
            side_effect=["div", {"id": "div1", "class": "container"}]
        )
        mock_element1.text_content = AsyncMock(return_value="Content 1")

        mock_element2.evaluate = AsyncMock(
            side_effect=["div", {"id": "div2", "class": "container"}]
        )
        mock_element2.text_content = AsyncMock(return_value="Content 2")

        tool_spec = AgentCoreBrowserToolSpec()
        tool_spec._session_manager = mock_session_manager

        result = await tool_spec.aget_elements(
            selector="div.container", thread_id="test-thread"
        )

        mock_session_manager.get_async_browser.assert_awaited_once_with("test-thread")
        mock_aget_current_page.assert_awaited_once_with(mock_browser)
        mock_page.query_selector_all.assert_awaited_once_with("div.container")
        mock_session_manager.release_async_browser.assert_awaited_once_with(
            "test-thread"
        )
        assert "Found 2 element(s) matching selector 'div.container'" in result
        assert '1. <div id="div1", class="container">Content 1</div>' in result
        assert '2. <div id="div2", class="container">Content 2</div>' in result

    @pytest.mark.asyncio
    @patch("llama_index.tools.aws_bedrock_agentcore.browser.base.aget_current_page")
    async def test_aget_elements_not_found(self, mock_aget_current_page):
        """Test aget_elements when no elements are found."""
        mock_session_manager = MagicMock()
        mock_browser = AsyncMock()
        mock_page = AsyncMock()

        mock_session_manager.get_async_browser = AsyncMock(return_value=mock_browser)
        mock_session_manager.release_async_browser = AsyncMock()
        mock_aget_current_page.return_value = mock_page
        mock_page.query_selector_all = AsyncMock(return_value=[])

        tool_spec = AgentCoreBrowserToolSpec()
        tool_spec._session_manager = mock_session_manager

        result = await tool_spec.aget_elements(
            selector="div.container", thread_id="test-thread"
        )

        mock_session_manager.get_async_browser.assert_awaited_once_with("test-thread")
        mock_aget_current_page.assert_awaited_once_with(mock_browser)
        mock_page.query_selector_all.assert_awaited_once_with("div.container")
        mock_session_manager.release_async_browser.assert_awaited_once_with(
            "test-thread"
        )
        assert "No elements found matching selector 'div.container'" in result

    @pytest.mark.asyncio
    @patch("llama_index.tools.aws_bedrock_agentcore.browser.base.aget_current_page")
    async def test_anavigate_back(self, mock_aget_current_page):
        """Test anavigate_back."""
        mock_session_manager = MagicMock()
        mock_browser = AsyncMock()
        mock_page = AsyncMock()
        mock_response = AsyncMock()

        mock_session_manager.get_async_browser = AsyncMock(return_value=mock_browser)
        mock_session_manager.release_async_browser = AsyncMock()
        mock_aget_current_page.return_value = mock_page
        mock_page.go_back = AsyncMock(return_value=mock_response)
        mock_page.url = "https://example.com/previous"

        tool_spec = AgentCoreBrowserToolSpec()
        tool_spec._session_manager = mock_session_manager

        result = await tool_spec.anavigate_back(thread_id="test-thread")

        mock_session_manager.get_async_browser.assert_awaited_once_with("test-thread")
        mock_aget_current_page.assert_awaited_once_with(mock_browser)
        mock_page.go_back.assert_awaited_once()
        mock_session_manager.release_async_browser.assert_awaited_once_with(
            "test-thread"
        )
        assert "Navigated back to https://example.com/previous" in result

    @pytest.mark.asyncio
    @patch("llama_index.tools.aws_bedrock_agentcore.browser.base.aget_current_page")
    async def test_anavigate_back_no_history(self, mock_aget_current_page):
        """Test anavigate_back when there's no history."""
        mock_session_manager = MagicMock()
        mock_browser = AsyncMock()
        mock_page = AsyncMock()

        mock_session_manager.get_async_browser = AsyncMock(return_value=mock_browser)
        mock_session_manager.release_async_browser = AsyncMock()
        mock_aget_current_page.return_value = mock_page
        mock_page.go_back = AsyncMock(return_value=None)

        tool_spec = AgentCoreBrowserToolSpec()
        tool_spec._session_manager = mock_session_manager

        result = await tool_spec.anavigate_back(thread_id="test-thread")

        mock_session_manager.get_async_browser.assert_awaited_once_with("test-thread")
        mock_aget_current_page.assert_awaited_once_with(mock_browser)
        mock_page.go_back.assert_awaited_once()
        mock_session_manager.release_async_browser.assert_awaited_once_with(
            "test-thread"
        )
        assert "Could not navigate back" in result

    @pytest.mark.asyncio
    @patch("llama_index.tools.aws_bedrock_agentcore.browser.base.aget_current_page")
    async def test_acurrent_webpage(self, mock_aget_current_page):
        """Test acurrent_webpage."""
        mock_session_manager = MagicMock()
        mock_browser = AsyncMock()
        mock_page = AsyncMock()

        mock_session_manager.get_async_browser = AsyncMock(return_value=mock_browser)
        mock_session_manager.release_async_browser = AsyncMock()
        mock_aget_current_page.return_value = mock_page
        mock_page.url = "https://example.com"
        mock_page.title = AsyncMock(return_value="Example Website")
        mock_page.evaluate = AsyncMock(
            return_value={
                "width": 1024,
                "height": 768,
                "links": 10,
                "images": 5,
                "forms": 2,
            }
        )

        tool_spec = AgentCoreBrowserToolSpec()
        tool_spec._session_manager = mock_session_manager

        result = await tool_spec.acurrent_webpage(thread_id="test-thread")

        mock_session_manager.get_async_browser.assert_awaited_once_with("test-thread")
        mock_aget_current_page.assert_awaited_once_with(mock_browser)
        mock_page.title.assert_awaited_once()
        mock_page.evaluate.assert_awaited_once()
        mock_session_manager.release_async_browser.assert_awaited_once_with(
            "test-thread"
        )
        assert "URL: https://example.com" in result
        assert "Title: Example Website" in result
        assert "Viewport size: 1024x768" in result
        assert "Links: 10" in result
        assert "Images: 5" in result
        assert "Forms: 2" in result

    @pytest.mark.asyncio
    async def test_cleanup_specific_thread(self):
        """Test cleanup for a specific thread."""
        mock_browser_client = MagicMock()

        tool_spec = AgentCoreBrowserToolSpec()
        tool_spec._browser_clients = {"test-thread": mock_browser_client}

        await tool_spec.cleanup(thread_id="test-thread")

        mock_browser_client.stop.assert_called_once()
        assert "test-thread" not in tool_spec._browser_clients

    @pytest.mark.asyncio
    async def test_cleanup_all_threads(self):
        """Test cleanup for all threads."""
        mock_browser_client1 = MagicMock()
        mock_browser_client2 = MagicMock()

        tool_spec = AgentCoreBrowserToolSpec()
        tool_spec._browser_clients = {
            "thread-1": mock_browser_client1,
            "thread-2": mock_browser_client2,
        }

        await tool_spec.cleanup()

        mock_browser_client1.stop.assert_called_once()
        mock_browser_client2.stop.assert_called_once()
        assert tool_spec._browser_clients == {}

    @pytest.mark.asyncio
    async def test_cleanup_with_exception(self):
        """Test cleanup when an exception occurs."""
        mock_browser_client = MagicMock()
        mock_browser_client.stop.side_effect = Exception("Test error")

        tool_spec = AgentCoreBrowserToolSpec()
        tool_spec._browser_clients = {"test-thread": mock_browser_client}

        # Should not raise an exception
        await tool_spec.cleanup(thread_id="test-thread")

        mock_browser_client.stop.assert_called_once()
        # Note: In the actual implementation, the thread is not removed from _browser_clients
        # when an exception occurs during stop(), so we don't assert that here
