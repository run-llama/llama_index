import pytest
from unittest.mock import patch, MagicMock

from llama_index.tools.aws_bedrock_agentcore.browser.browser_session_manager import (
    BrowserSessionManager,
)


class TestBrowserSessionManager:
    def test_init(self):
        manager = BrowserSessionManager(region="us-east-1")
        assert manager.region == "us-east-1"
        assert manager._async_sessions == {}
        assert manager._sync_sessions == {}

    def test_init_default_region(self):
        manager = BrowserSessionManager()
        assert manager.region == "us-west-2"
        assert manager._async_sessions == {}
        assert manager._sync_sessions == {}

    @patch(
        "llama_index.tools.aws_bedrock_agentcore.browser.browser_session_manager.BrowserClient"
    )
    def test_get_sync_browser_existing(self, mock_browser_client):
        manager = BrowserSessionManager()
        mock_browser = MagicMock()
        mock_client = MagicMock()
        manager._sync_sessions = {"test-thread": (mock_client, mock_browser, False)}

        browser = manager.get_sync_browser("test-thread")

        assert browser == mock_browser
        assert manager._sync_sessions["test-thread"] == (
            mock_client,
            mock_browser,
            True,
        )

    def test_get_sync_browser_in_use(self):
        manager = BrowserSessionManager()
        mock_browser = MagicMock()
        mock_client = MagicMock()
        manager._sync_sessions = {"test-thread": (mock_client, mock_browser, True)}

        with pytest.raises(
            RuntimeError,
            match="Browser session for thread test-thread is already in use",
        ):
            manager.get_sync_browser("test-thread")

    @patch.object(BrowserSessionManager, "_create_sync_browser_session")
    def test_get_sync_browser_new(self, mock_create_sync_browser):
        mock_browser = MagicMock()
        mock_create_sync_browser.return_value = mock_browser

        manager = BrowserSessionManager()
        browser = manager.get_sync_browser("test-thread")

        assert browser == mock_browser
        mock_create_sync_browser.assert_called_once_with("test-thread")

    def test_release_sync_browser(self):
        manager = BrowserSessionManager()
        mock_browser = MagicMock()
        mock_client = MagicMock()
        manager._sync_sessions = {"test-thread": (mock_client, mock_browser, True)}

        manager.release_sync_browser("test-thread")

        assert manager._sync_sessions["test-thread"] == (
            mock_client,
            mock_browser,
            False,
        )

    def test_release_sync_browser_not_found(self):
        manager = BrowserSessionManager()
        # Should not raise an exception
        manager.release_sync_browser("test-thread")

    def test_close_sync_browser(self):
        manager = BrowserSessionManager()
        mock_browser = MagicMock()
        mock_client = MagicMock()
        manager._sync_sessions = {"test-thread": (mock_client, mock_browser, False)}

        manager.close_sync_browser("test-thread")

        mock_browser.close.assert_called_once()
        mock_client.stop.assert_called_once()
        assert "test-thread" not in manager._sync_sessions

    def test_close_sync_browser_not_found(self):
        manager = BrowserSessionManager()
        # Should not raise an exception
        manager.close_sync_browser("test-thread")

    def test_close_sync_browser_with_errors(self):
        manager = BrowserSessionManager()
        mock_browser = MagicMock()
        mock_browser.close.side_effect = Exception("Browser close error")
        mock_client = MagicMock()
        mock_client.stop.side_effect = Exception("Client stop error")
        manager._sync_sessions = {"test-thread": (mock_client, mock_browser, False)}

        # Should not raise an exception
        manager.close_sync_browser("test-thread")

        mock_browser.close.assert_called_once()
        mock_client.stop.assert_called_once()
        assert "test-thread" not in manager._sync_sessions
