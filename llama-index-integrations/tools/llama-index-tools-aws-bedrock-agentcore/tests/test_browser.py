import os
from unittest.mock import patch, MagicMock

from llama_index.tools.aws_bedrock_agentcore import AgentCoreBrowserToolSpec
from llama_index.tools.aws_bedrock_agentcore.browser.base import get_aws_region
from llama_index.tools.aws_bedrock_agentcore.browser.utils import get_current_page


class TestGetAwsRegion:
    @patch.dict(os.environ, {"AWS_REGION": "us-east-1"})
    def test_get_aws_region_from_aws_region(self):
        assert get_aws_region() == "us-east-1"

    @patch.dict(
        os.environ, {"AWS_DEFAULT_REGION": "us-west-1", "AWS_REGION": ""}, clear=True
    )
    def test_get_aws_region_from_aws_default_region(self):
        assert get_aws_region() == "us-west-1"

    @patch.dict(os.environ, {}, clear=True)
    def test_get_aws_region_default(self):
        assert get_aws_region() == "us-west-2"


class TestBrowserUtils:
    def test_get_current_page_no_contexts(self):
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

    def test_get_current_page_with_context_no_pages(self):
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

    def test_get_current_page_with_context_and_pages(self):
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page1 = MagicMock()
        mock_page2 = MagicMock()
        mock_context.pages = [mock_page1, mock_page2]
        mock_browser.contexts = [mock_context]

        result = get_current_page(mock_browser)

        assert result == mock_page2
        mock_browser.new_context.assert_not_called()
        mock_context.new_page.assert_not_called()


class TestAgentCoreBrowserToolSpec:
    @patch("llama_index.tools.aws_bedrock_agentcore.browser.base.BrowserSessionManager")
    def test_init(self, mock_browser_session_manager):
        tool_spec = AgentCoreBrowserToolSpec(region="us-east-1")
        assert tool_spec.region == "us-east-1"
        assert tool_spec._browser_clients == {}
        mock_browser_session_manager.assert_called_once_with(region="us-east-1")

    @patch("llama_index.tools.aws_bedrock_agentcore.browser.base.get_aws_region")
    @patch("llama_index.tools.aws_bedrock_agentcore.browser.base.BrowserSessionManager")
    def test_init_default_region(
        self, mock_browser_session_manager, mock_get_aws_region
    ):
        mock_get_aws_region.return_value = "us-west-2"
        tool_spec = AgentCoreBrowserToolSpec()
        assert tool_spec.region == "us-west-2"
        mock_get_aws_region.assert_called_once()
        mock_browser_session_manager.assert_called_once_with(region="us-west-2")

    @patch("llama_index.tools.aws_bedrock_agentcore.browser.base.BrowserClient")
    def test_get_or_create_browser_client_new(self, mock_browser_client):
        mock_instance = MagicMock()
        mock_browser_client.return_value = mock_instance

        tool_spec = AgentCoreBrowserToolSpec(region="us-east-1")
        client = tool_spec._get_or_create_browser_client("test-thread")

        assert client == mock_instance
        assert "test-thread" in tool_spec._browser_clients
        assert tool_spec._browser_clients["test-thread"] == mock_instance

        mock_browser_client.assert_called_once_with("us-east-1")

    @patch("llama_index.tools.aws_bedrock_agentcore.browser.base.BrowserClient")
    def test_get_or_create_browser_client_existing(self, mock_browser_client):
        mock_instance = MagicMock()

        tool_spec = AgentCoreBrowserToolSpec(region="us-east-1")
        tool_spec._browser_clients["test-thread"] = mock_instance

        client = tool_spec._get_or_create_browser_client("test-thread")

        assert client == mock_instance
        mock_browser_client.assert_not_called()

    def test_navigate_browser_invalid_url(self):
        tool_spec = AgentCoreBrowserToolSpec()

        result = tool_spec.navigate_browser(
            url="ftp://example.com", thread_id="test-thread"
        )

        assert "URL scheme must be 'http' or 'https'" in result

    @patch("llama_index.tools.aws_bedrock_agentcore.browser.base.get_current_page")
    def test_navigate_browser(self, mock_get_current_page):
        mock_session_manager = MagicMock()
        mock_browser = MagicMock()
        mock_page = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200

        mock_session_manager.get_sync_browser.return_value = mock_browser
        mock_get_current_page.return_value = mock_page
        mock_page.goto.return_value = mock_response

        tool_spec = AgentCoreBrowserToolSpec()
        tool_spec._session_manager = mock_session_manager

        result = tool_spec.navigate_browser(
            url="https://example.com", thread_id="test-thread"
        )

        mock_session_manager.get_sync_browser.assert_called_once_with("test-thread")
        mock_get_current_page.assert_called_once_with(mock_browser)
        mock_page.goto.assert_called_once_with("https://example.com")
        mock_session_manager.release_sync_browser.assert_called_once_with("test-thread")
        assert "Navigated to https://example.com with status code 200" in result

    def test_navigate_browser_exception(self):
        mock_session_manager = MagicMock()
        mock_session_manager.get_sync_browser.side_effect = Exception("Test error")

        tool_spec = AgentCoreBrowserToolSpec()
        tool_spec._session_manager = mock_session_manager

        result = tool_spec.navigate_browser(
            url="https://example.com", thread_id="test-thread"
        )

        assert "Error navigating to URL: Test error" in result

    @patch("llama_index.tools.aws_bedrock_agentcore.browser.base.get_current_page")
    def test_click_element(self, mock_get_current_page):
        mock_session_manager = MagicMock()
        mock_browser = MagicMock()
        mock_page = MagicMock()

        mock_session_manager.get_sync_browser.return_value = mock_browser
        mock_get_current_page.return_value = mock_page

        tool_spec = AgentCoreBrowserToolSpec()
        tool_spec._session_manager = mock_session_manager

        result = tool_spec.click_element(selector="#button", thread_id="test-thread")

        mock_session_manager.get_sync_browser.assert_called_once_with("test-thread")
        mock_get_current_page.assert_called_once_with(mock_browser)
        mock_page.click.assert_called_once_with("#button", timeout=5000)
        mock_session_manager.release_sync_browser.assert_called_once_with("test-thread")
        assert "Clicked on element with selector '#button'" in result

    @patch("llama_index.tools.aws_bedrock_agentcore.browser.base.get_current_page")
    def test_click_element_not_found(self, mock_get_current_page):
        mock_session_manager = MagicMock()
        mock_browser = MagicMock()
        mock_page = MagicMock()

        mock_session_manager.get_sync_browser.return_value = mock_browser
        mock_get_current_page.return_value = mock_page
        mock_page.click.side_effect = Exception("Element not found")

        tool_spec = AgentCoreBrowserToolSpec()
        tool_spec._session_manager = mock_session_manager

        result = tool_spec.click_element(selector="#button", thread_id="test-thread")

        mock_session_manager.get_sync_browser.assert_called_once_with("test-thread")
        mock_get_current_page.assert_called_once_with(mock_browser)
        mock_page.click.assert_called_once_with("#button", timeout=5000)
        mock_session_manager.release_sync_browser.assert_called_once_with("test-thread")
        assert "Unable to click on element with selector '#button'" in result

    @patch("llama_index.tools.aws_bedrock_agentcore.browser.base.get_current_page")
    def test_extract_text_whole_page(self, mock_get_current_page):
        mock_session_manager = MagicMock()
        mock_browser = MagicMock()
        mock_page = MagicMock()

        mock_session_manager.get_sync_browser.return_value = mock_browser
        mock_get_current_page.return_value = mock_page
        mock_page.content.return_value = "<html><body>Hello World</body></html>"

        tool_spec = AgentCoreBrowserToolSpec()
        tool_spec._session_manager = mock_session_manager

        result = tool_spec.extract_text(thread_id="test-thread")

        mock_session_manager.get_sync_browser.assert_called_once_with("test-thread")
        mock_get_current_page.assert_called_once_with(mock_browser)
        mock_page.content.assert_called_once()
        mock_session_manager.release_sync_browser.assert_called_once_with("test-thread")
        assert result == "<html><body>Hello World</body></html>"

    @patch("llama_index.tools.aws_bedrock_agentcore.browser.base.get_current_page")
    def test_extract_text_with_selector(self, mock_get_current_page):
        mock_session_manager = MagicMock()
        mock_browser = MagicMock()
        mock_page = MagicMock()
        mock_element = MagicMock()

        mock_session_manager.get_sync_browser.return_value = mock_browser
        mock_get_current_page.return_value = mock_page
        mock_page.query_selector.return_value = mock_element
        mock_element.text_content.return_value = "Hello World"

        tool_spec = AgentCoreBrowserToolSpec()
        tool_spec._session_manager = mock_session_manager

        result = tool_spec.extract_text(selector="#content", thread_id="test-thread")

        mock_session_manager.get_sync_browser.assert_called_once_with("test-thread")
        mock_get_current_page.assert_called_once_with(mock_browser)
        mock_page.query_selector.assert_called_once_with("#content")
        mock_element.text_content.assert_called_once()
        mock_session_manager.release_sync_browser.assert_called_once_with("test-thread")
        assert result == "Hello World"

    @patch("llama_index.tools.aws_bedrock_agentcore.browser.base.get_current_page")
    def test_extract_text_selector_not_found(self, mock_get_current_page):
        mock_session_manager = MagicMock()
        mock_browser = MagicMock()
        mock_page = MagicMock()

        mock_session_manager.get_sync_browser.return_value = mock_browser
        mock_get_current_page.return_value = mock_page
        mock_page.query_selector.return_value = None

        tool_spec = AgentCoreBrowserToolSpec()
        tool_spec._session_manager = mock_session_manager

        result = tool_spec.extract_text(selector="#content", thread_id="test-thread")

        mock_session_manager.get_sync_browser.assert_called_once_with("test-thread")
        mock_get_current_page.assert_called_once_with(mock_browser)
        mock_page.query_selector.assert_called_once_with("#content")
        mock_session_manager.release_sync_browser.assert_called_once_with("test-thread")
        assert "No element found with selector '#content'" in result

    @patch("llama_index.tools.aws_bedrock_agentcore.browser.base.get_current_page")
    def test_extract_hyperlinks(self, mock_get_current_page):
        mock_session_manager = MagicMock()
        mock_browser = MagicMock()
        mock_page = MagicMock()

        mock_session_manager.get_sync_browser.return_value = mock_browser
        mock_get_current_page.return_value = mock_page
        mock_page.eval_on_selector_all.return_value = [
            {"text": "Link 1", "href": "https://example.com/1"},
            {"text": "Link 2", "href": "https://example.com/2"},
        ]

        tool_spec = AgentCoreBrowserToolSpec()
        tool_spec._session_manager = mock_session_manager

        result = tool_spec.extract_hyperlinks(thread_id="test-thread")

        mock_session_manager.get_sync_browser.assert_called_once_with("test-thread")
        mock_get_current_page.assert_called_once_with(mock_browser)
        mock_page.eval_on_selector_all.assert_called_once()
        mock_session_manager.release_sync_browser.assert_called_once_with("test-thread")
        assert "1. Link 1: https://example.com/1" in result
        assert "2. Link 2: https://example.com/2" in result

    @patch("llama_index.tools.aws_bedrock_agentcore.browser.base.get_current_page")
    def test_extract_hyperlinks_no_links(self, mock_get_current_page):
        mock_session_manager = MagicMock()
        mock_browser = MagicMock()
        mock_page = MagicMock()

        mock_session_manager.get_sync_browser.return_value = mock_browser
        mock_get_current_page.return_value = mock_page
        mock_page.eval_on_selector_all.return_value = []

        tool_spec = AgentCoreBrowserToolSpec()
        tool_spec._session_manager = mock_session_manager

        result = tool_spec.extract_hyperlinks(thread_id="test-thread")

        mock_session_manager.get_sync_browser.assert_called_once_with("test-thread")
        mock_get_current_page.assert_called_once_with(mock_browser)
        mock_page.eval_on_selector_all.assert_called_once()
        mock_session_manager.release_sync_browser.assert_called_once_with("test-thread")
        assert "No hyperlinks found on the page" in result

    @patch("llama_index.tools.aws_bedrock_agentcore.browser.base.get_current_page")
    def test_get_elements(self, mock_get_current_page):
        mock_session_manager = MagicMock()
        mock_browser = MagicMock()
        mock_page = MagicMock()
        mock_element1 = MagicMock()
        mock_element2 = MagicMock()

        mock_session_manager.get_sync_browser.return_value = mock_browser
        mock_get_current_page.return_value = mock_page
        mock_page.query_selector_all.return_value = [mock_element1, mock_element2]

        mock_element1.evaluate.side_effect = [
            "div",
            {"id": "div1", "class": "container"},
        ]
        mock_element1.text_content.return_value = "Content 1"

        mock_element2.evaluate.side_effect = [
            "div",
            {"id": "div2", "class": "container"},
        ]
        mock_element2.text_content.return_value = "Content 2"

        tool_spec = AgentCoreBrowserToolSpec()
        tool_spec._session_manager = mock_session_manager

        result = tool_spec.get_elements(
            selector="div.container", thread_id="test-thread"
        )

        mock_session_manager.get_sync_browser.assert_called_once_with("test-thread")
        mock_get_current_page.assert_called_once_with(mock_browser)
        mock_page.query_selector_all.assert_called_once_with("div.container")
        mock_session_manager.release_sync_browser.assert_called_once_with("test-thread")
        assert "Found 2 element(s) matching selector 'div.container'" in result
        assert '1. <div id="div1", class="container">Content 1</div>' in result
        assert '2. <div id="div2", class="container">Content 2</div>' in result

    @patch("llama_index.tools.aws_bedrock_agentcore.browser.base.get_current_page")
    def test_get_elements_not_found(self, mock_get_current_page):
        mock_session_manager = MagicMock()
        mock_browser = MagicMock()
        mock_page = MagicMock()

        mock_session_manager.get_sync_browser.return_value = mock_browser
        mock_get_current_page.return_value = mock_page
        mock_page.query_selector_all.return_value = []

        tool_spec = AgentCoreBrowserToolSpec()
        tool_spec._session_manager = mock_session_manager

        result = tool_spec.get_elements(
            selector="div.container", thread_id="test-thread"
        )

        mock_session_manager.get_sync_browser.assert_called_once_with("test-thread")
        mock_get_current_page.assert_called_once_with(mock_browser)
        mock_page.query_selector_all.assert_called_once_with("div.container")
        mock_session_manager.release_sync_browser.assert_called_once_with("test-thread")
        assert "No elements found matching selector 'div.container'" in result

    @patch("llama_index.tools.aws_bedrock_agentcore.browser.base.get_current_page")
    def test_navigate_back(self, mock_get_current_page):
        mock_session_manager = MagicMock()
        mock_browser = MagicMock()
        mock_page = MagicMock()
        mock_response = MagicMock()

        mock_session_manager.get_sync_browser.return_value = mock_browser
        mock_get_current_page.return_value = mock_page
        mock_page.go_back.return_value = mock_response
        mock_page.url = "https://example.com/previous"

        tool_spec = AgentCoreBrowserToolSpec()
        tool_spec._session_manager = mock_session_manager

        result = tool_spec.navigate_back(thread_id="test-thread")

        mock_session_manager.get_sync_browser.assert_called_once_with("test-thread")
        mock_get_current_page.assert_called_once_with(mock_browser)
        mock_page.go_back.assert_called_once()
        mock_session_manager.release_sync_browser.assert_called_once_with("test-thread")
        assert "Navigated back to https://example.com/previous" in result

    @patch("llama_index.tools.aws_bedrock_agentcore.browser.base.get_current_page")
    def test_navigate_back_no_history(self, mock_get_current_page):
        mock_session_manager = MagicMock()
        mock_browser = MagicMock()
        mock_page = MagicMock()

        mock_session_manager.get_sync_browser.return_value = mock_browser
        mock_get_current_page.return_value = mock_page
        mock_page.go_back.return_value = None

        tool_spec = AgentCoreBrowserToolSpec()
        tool_spec._session_manager = mock_session_manager

        result = tool_spec.navigate_back(thread_id="test-thread")

        mock_session_manager.get_sync_browser.assert_called_once_with("test-thread")
        mock_get_current_page.assert_called_once_with(mock_browser)
        mock_page.go_back.assert_called_once()
        mock_session_manager.release_sync_browser.assert_called_once_with("test-thread")
        assert "Could not navigate back" in result

    @patch("llama_index.tools.aws_bedrock_agentcore.browser.base.get_current_page")
    def test_current_webpage(self, mock_get_current_page):
        mock_session_manager = MagicMock()
        mock_browser = MagicMock()
        mock_page = MagicMock()

        mock_session_manager.get_sync_browser.return_value = mock_browser
        mock_get_current_page.return_value = mock_page
        mock_page.url = "https://example.com"
        mock_page.title.return_value = "Example Website"
        mock_page.evaluate.return_value = {
            "width": 1024,
            "height": 768,
            "links": 10,
            "images": 5,
            "forms": 2,
        }

        tool_spec = AgentCoreBrowserToolSpec()
        tool_spec._session_manager = mock_session_manager

        result = tool_spec.current_webpage(thread_id="test-thread")

        mock_session_manager.get_sync_browser.assert_called_once_with("test-thread")
        mock_get_current_page.assert_called_once_with(mock_browser)
        mock_page.title.assert_called_once()
        mock_page.evaluate.assert_called_once()
        mock_session_manager.release_sync_browser.assert_called_once_with("test-thread")
        assert "URL: https://example.com" in result
        assert "Title: Example Website" in result
        assert "Viewport size: 1024x768" in result
        assert "Links: 10" in result
        assert "Images: 5" in result
        assert "Forms: 2" in result

    def test_cleanup_thread(self):
        mock_browser_client = MagicMock()

        tool_spec = AgentCoreBrowserToolSpec()
        tool_spec._browser_clients = {"test-thread": mock_browser_client}

        # Call cleanup synchronously for testing
        tool_spec._browser_clients["test-thread"].stop = MagicMock()

        # Simulate cleanup
        tool_spec._browser_clients["test-thread"].stop()
        del tool_spec._browser_clients["test-thread"]

        mock_browser_client.stop.assert_called_once()
        assert "test-thread" not in tool_spec._browser_clients
