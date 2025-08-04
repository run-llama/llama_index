from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Tuple

from bedrock_agentcore.tools.browser_client import BrowserClient

if TYPE_CHECKING:
    from playwright.async_api import Browser as AsyncBrowser
    from playwright.sync_api import Browser as SyncBrowser

logger = logging.getLogger(__name__)


class BrowserSessionManager:
    """
    Manages browser sessions for different threads.

    This class maintains separate browser sessions for different threads,
    enabling concurrent usage of browsers in multi-threaded environments.
    Browsers are created lazily only when needed by tools.

    Concurrency protection is also implemented. Each browser session is tied
    to a specific thread_id and includes protection against concurrent usage.
    When a browser is obtained via get_async_browser() or get_sync_browser(),
    it is marked as "in use", and subsequent attempts to access the same
    browser session will raise a RuntimeError until it is released.
    """

    def __init__(self, region: str = "us-west-2"):
        """
        Initialize the browser session manager.

        Args:
            region: AWS region for browser client

        """
        self.region = region
        self._async_sessions: Dict[str, Tuple[BrowserClient, AsyncBrowser, bool]] = {}
        self._sync_sessions: Dict[str, Tuple[BrowserClient, SyncBrowser, bool]] = {}

    async def get_async_browser(self, thread_id: str) -> AsyncBrowser:
        """
        Get or create an async browser for the specified thread.

        Args:
            thread_id: Unique identifier for the thread requesting the browser

        Returns:
            An async browser instance specific to the thread

        Raises:
            RuntimeError: If the browser session is already in use by another caller

        """
        if thread_id in self._async_sessions:
            client, browser, in_use = self._async_sessions[thread_id]
            if in_use:
                raise RuntimeError(
                    f"Browser session for thread {thread_id} is already in use. "
                    "Use a different thread_id for concurrent operations."
                )
            self._async_sessions[thread_id] = (client, browser, True)
            return browser

        return await self._create_async_browser_session(thread_id)

    def get_sync_browser(self, thread_id: str) -> SyncBrowser:
        """
        Get or create a sync browser for the specified thread.

        Args:
            thread_id: Unique identifier for the thread requesting the browser

        Returns:
            A sync browser instance specific to the thread

        Raises:
            RuntimeError: If the browser session is already in use by another caller

        """
        if thread_id in self._sync_sessions:
            client, browser, in_use = self._sync_sessions[thread_id]
            if in_use:
                raise RuntimeError(
                    f"Browser session for thread {thread_id} is already in use. "
                    "Use a different thread_id for concurrent operations."
                )
            self._sync_sessions[thread_id] = (client, browser, True)
            return browser

        return self._create_sync_browser_session(thread_id)

    async def _create_async_browser_session(self, thread_id: str) -> AsyncBrowser:
        """
        Create a new async browser session for the specified thread.

        Args:
            thread_id: Unique identifier for the thread

        Returns:
            The newly created async browser instance

        Raises:
            Exception: If browser session creation fails

        """
        browser_client = BrowserClient(region=self.region)

        try:
            # Start browser session
            browser_client.start()

            # Get WebSocket connection info
            ws_url, headers = browser_client.generate_ws_headers()

            logger.info(
                f"Connecting to async WebSocket endpoint for thread {thread_id}: {ws_url}"
            )

            from playwright.async_api import async_playwright

            # Connect to browser using Playwright
            playwright = await async_playwright().start()
            browser = await playwright.chromium.connect_over_cdp(
                endpoint_url=ws_url, headers=headers, timeout=30000
            )
            logger.info(
                f"Successfully connected to async browser for thread {thread_id}"
            )

            # Store session resources
            self._async_sessions[thread_id] = (browser_client, browser, True)

            return browser

        except Exception as e:
            logger.error(
                f"Failed to create async browser session for thread {thread_id}: {e}"
            )

            # Clean up resources if session creation fails
            if browser_client:
                try:
                    browser_client.stop()
                except Exception as cleanup_error:
                    logger.warning(f"Error cleaning up browser client: {cleanup_error}")

            raise

    def _create_sync_browser_session(self, thread_id: str) -> SyncBrowser:
        """
        Create a new sync browser session for the specified thread.

        Args:
            thread_id: Unique identifier for the thread

        Returns:
            The newly created sync browser instance

        Raises:
            Exception: If browser session creation fails

        """
        browser_client = BrowserClient(region=self.region)

        try:
            # Start browser session
            browser_client.start()

            # Get WebSocket connection info
            ws_url, headers = browser_client.generate_ws_headers()

            logger.info(
                f"Connecting to sync WebSocket endpoint for thread {thread_id}: {ws_url}"
            )

            from playwright.sync_api import sync_playwright

            # Connect to browser using Playwright
            playwright = sync_playwright().start()
            browser = playwright.chromium.connect_over_cdp(
                endpoint_url=ws_url, headers=headers, timeout=30000
            )
            logger.info(
                f"Successfully connected to sync browser for thread {thread_id}"
            )

            # Store session resources
            self._sync_sessions[thread_id] = (browser_client, browser, True)

            return browser

        except Exception as e:
            logger.error(
                f"Failed to create sync browser session for thread {thread_id}: {e}"
            )

            # Clean up resources if session creation fails
            if browser_client:
                try:
                    browser_client.stop()
                except Exception as cleanup_error:
                    logger.warning(f"Error cleaning up browser client: {cleanup_error}")

            raise

    async def release_async_browser(self, thread_id: str) -> None:
        """
        Release the async browser session for the specified thread.

        Args:
            thread_id: Unique identifier for the thread

        """
        if thread_id not in self._async_sessions:
            logger.warning(f"No async browser session found for thread {thread_id}")
            return

        client, browser, in_use = self._async_sessions[thread_id]
        if in_use:
            self._async_sessions[thread_id] = (client, browser, False)
            logger.info(f"Released async browser for thread {thread_id}")

    def release_sync_browser(self, thread_id: str) -> None:
        """
        Release the sync browser session for the specified thread.

        Args:
            thread_id: Unique identifier for the thread

        """
        if thread_id not in self._sync_sessions:
            logger.warning(f"No sync browser session found for thread {thread_id}")
            return

        client, browser, in_use = self._sync_sessions[thread_id]
        if in_use:
            self._sync_sessions[thread_id] = (client, browser, False)
            logger.info(f"Released sync browser for thread {thread_id}")

    async def close_async_browser(self, thread_id: str) -> None:
        """
        Close the async browser session for the specified thread.

        Args:
            thread_id: Unique identifier for the thread

        """
        if thread_id not in self._async_sessions:
            logger.warning(f"No async browser session found for thread {thread_id}")
            return

        client, browser, _ = self._async_sessions[thread_id]

        # Close browser
        if browser:
            try:
                await browser.close()
            except Exception as e:
                logger.warning(
                    f"Error closing async browser for thread {thread_id}: {e}"
                )

        # Stop browser client
        if client:
            try:
                client.stop()
            except Exception as e:
                logger.warning(
                    f"Error stopping browser client for thread {thread_id}: {e}"
                )

        # Remove session from dictionary
        del self._async_sessions[thread_id]
        logger.info(f"Async browser session cleaned up for thread {thread_id}")

    def close_sync_browser(self, thread_id: str) -> None:
        """
        Close the sync browser session for the specified thread.

        Args:
            thread_id: Unique identifier for the thread

        """
        if thread_id not in self._sync_sessions:
            logger.warning(f"No sync browser session found for thread {thread_id}")
            return

        client, browser, _ = self._sync_sessions[thread_id]

        # Close browser
        if browser:
            try:
                browser.close()
            except Exception as e:
                logger.warning(
                    f"Error closing sync browser for thread {thread_id}: {e}"
                )

        # Stop browser client
        if client:
            try:
                client.stop()
            except Exception as e:
                logger.warning(
                    f"Error stopping browser client for thread {thread_id}: {e}"
                )

        # Remove session from dictionary
        del self._sync_sessions[thread_id]
        logger.info(f"Sync browser session cleaned up for thread {thread_id}")

    async def close_all_browsers(self) -> None:
        """Close all browser sessions."""
        # Close all async browsers
        async_thread_ids = list(self._async_sessions.keys())
        for thread_id in async_thread_ids:
            await self.close_async_browser(thread_id)

        # Close all sync browsers
        sync_thread_ids = list(self._sync_sessions.keys())
        for thread_id in sync_thread_ids:
            self.close_sync_browser(thread_id)

        logger.info("All browser sessions closed")
