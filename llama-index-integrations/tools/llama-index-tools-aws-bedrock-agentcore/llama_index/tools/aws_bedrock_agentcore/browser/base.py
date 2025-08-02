import os
import logging
from typing import Dict, Optional
from urllib.parse import urlparse

from llama_index.core.tools.tool_spec.base import BaseToolSpec

from bedrock_agentcore.tools.browser_client import BrowserClient

from .browser_session_manager import BrowserSessionManager
from .utils import aget_current_page, get_current_page

DEFAULT_BROWSER_IDENTIFIER = "aws.browser.v1"
DEFAULT_BROWSER_SESSION_TIMEOUT = 3600
DEFAULT_BROWSER_LIVE_VIEW_PRESIGNED_URL_TIMEOUT = 300

logger = logging.getLogger(__name__)


def get_aws_region() -> str:
    """Get the AWS region from environment variables or use default."""
    return os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-west-2"


class AgentCoreBrowserToolSpec(BaseToolSpec):
    """
    AWS Bedrock AgentCore Browser Tool Spec.

    This toolkit provides a set of tools for working with a remote browser environment:

    * navigate_browser - Navigate to a URL
    * click_element - Click on an element using CSS selectors
    * extract_text - Extract all text from the current webpage
    * extract_hyperlinks - Extract all hyperlinks from the current webpage
    * get_elements - Get elements matching a CSS selector
    * navigate_back - Navigate to the previous page
    * current_webpage - Get information about the current webpage

    The toolkit supports multiple threads by maintaining separate browser sessions for each thread ID.
    """

    spec_functions = [
        ("navigate_browser", "anavigate_browser"),
        ("click_element", "aclick_element"),
        ("extract_text", "aextract_text"),
        ("extract_hyperlinks", "aextract_hyperlinks"),
        ("get_elements", "aget_elements"),
        ("navigate_back", "anavigate_back"),
        ("current_webpage", "acurrent_webpage"),
    ]

    def __init__(self, region: Optional[str] = None) -> None:
        """
        Initialize the AWS Bedrock AgentCore Browser Tool Spec.

        Args:
            region (Optional[str]): AWS region to use for Bedrock AgentCore services.
                If not provided, will try to get it from environment variables.

        """
        self.region = region if region is not None else get_aws_region()
        self._browser_clients: Dict[str, BrowserClient] = {}
        self._session_manager = BrowserSessionManager(region=self.region)

    def _get_or_create_browser_client(
        self, thread_id: str = "default"
    ) -> BrowserClient:
        """
        Get or create a browser client for the specified thread.

        Args:
            thread_id: Thread ID for the browser session

        Returns:
            BrowserClient instance

        """
        if thread_id in self._browser_clients:
            return self._browser_clients[thread_id]

        # Create a new browser client for this thread
        browser_client = BrowserClient(self.region)
        self._browser_clients[thread_id] = browser_client
        return browser_client

    def navigate_browser(
        self,
        url: str,
        thread_id: str = "default",
    ) -> str:
        """
        Navigate to a URL (synchronous version).

        Args:
            url (str): URL to navigate to.
            thread_id (str): Thread ID for the browser session.

        Returns:
            str: Confirmation message.

        """
        try:
            # Validate URL scheme
            parsed_url = urlparse(url)
            if parsed_url.scheme not in ("http", "https"):
                return f"URL scheme must be 'http' or 'https', got: {parsed_url.scheme}"

            # Get browser and navigate to URL
            browser = self._session_manager.get_sync_browser(thread_id)
            page = get_current_page(browser)
            response = page.goto(url)
            status = response.status if response else "unknown"

            # Release the browser
            self._session_manager.release_sync_browser(thread_id)

            return f"Navigated to {url} with status code {status}"
        except Exception as e:
            return f"Error navigating to URL: {e!s}"

    async def anavigate_browser(
        self,
        url: str,
        thread_id: str = "default",
    ) -> str:
        """
        Navigate to a URL (asynchronous version).

        Args:
            url (str): URL to navigate to.
            thread_id (str): Thread ID for the browser session.

        Returns:
            str: Confirmation message.

        """
        try:
            # Validate URL scheme
            parsed_url = urlparse(url)
            if parsed_url.scheme not in ("http", "https"):
                return f"URL scheme must be 'http' or 'https', got: {parsed_url.scheme}"

            # Get browser and navigate to URL
            browser = await self._session_manager.get_async_browser(thread_id)
            page = await aget_current_page(browser)
            response = await page.goto(url)
            status = response.status if response else "unknown"

            # Release the browser
            await self._session_manager.release_async_browser(thread_id)

            return f"Navigated to {url} with status code {status}"
        except Exception as e:
            return f"Error navigating to URL: {e!s}"

    def click_element(
        self,
        selector: str,
        thread_id: str = "default",
    ) -> str:
        """
        Click on an element with the given CSS selector (synchronous version).

        Args:
            selector (str): CSS selector for the element to click on.
            thread_id (str): Thread ID for the browser session.

        Returns:
            str: Confirmation message.

        """
        try:
            # Get browser and click on element
            browser = self._session_manager.get_sync_browser(thread_id)
            page = get_current_page(browser)

            try:
                page.click(selector, timeout=5000)
                result = f"Clicked on element with selector '{selector}'"
            except Exception as click_error:
                result = f"Unable to click on element with selector '{selector}': {click_error!s}"

            # Release the browser
            self._session_manager.release_sync_browser(thread_id)

            return result
        except Exception as e:
            return f"Error clicking on element: {e!s}"

    async def aclick_element(
        self,
        selector: str,
        thread_id: str = "default",
    ) -> str:
        """
        Click on an element with the given CSS selector (asynchronous version).

        Args:
            selector (str): CSS selector for the element to click on.
            thread_id (str): Thread ID for the browser session.

        Returns:
            str: Confirmation message.

        """
        try:
            # Get browser and click on element
            browser = await self._session_manager.get_async_browser(thread_id)
            page = await aget_current_page(browser)

            try:
                await page.click(selector, timeout=5000)
                result = f"Clicked on element with selector '{selector}'"
            except Exception as click_error:
                result = f"Unable to click on element with selector '{selector}': {click_error!s}"

            # Release the browser
            await self._session_manager.release_async_browser(thread_id)

            return result
        except Exception as e:
            return f"Error clicking on element: {e!s}"

    def extract_text(
        self,
        selector: Optional[str] = None,
        thread_id: str = "default",
    ) -> str:
        """
        Extract text from the current page (synchronous version).

        Args:
            selector (Optional[str]): CSS selector for the element to extract text from. If not provided, extracts text from the entire page.
            thread_id (str): Thread ID for the browser session.

        Returns:
            str: The extracted text.

        """
        try:
            # Get browser and extract text
            browser = self._session_manager.get_sync_browser(thread_id)
            page = get_current_page(browser)

            if selector:
                try:
                    element = page.query_selector(selector)
                    if element:
                        text = element.text_content()
                        result = text if text else "Element found but contains no text"
                    else:
                        result = f"No element found with selector '{selector}'"
                except Exception as selector_error:
                    result = f"Error extracting text from selector '{selector}': {selector_error!s}"
            else:
                # Extract text from the entire page
                result = page.content()

            # Release the browser
            self._session_manager.release_sync_browser(thread_id)

            return result
        except Exception as e:
            return f"Error extracting text: {e!s}"

    async def aextract_text(
        self,
        selector: Optional[str] = None,
        thread_id: str = "default",
    ) -> str:
        """
        Extract text from the current page (asynchronous version).

        Args:
            selector (Optional[str]): CSS selector for the element to extract text from. If not provided, extracts text from the entire page.
            thread_id (str): Thread ID for the browser session.

        Returns:
            str: The extracted text.

        """
        try:
            # Get browser and extract text
            browser = await self._session_manager.get_async_browser(thread_id)
            page = await aget_current_page(browser)

            if selector:
                try:
                    element = await page.query_selector(selector)
                    if element:
                        text = await element.text_content()
                        result = text if text else "Element found but contains no text"
                    else:
                        result = f"No element found with selector '{selector}'"
                except Exception as selector_error:
                    result = f"Error extracting text from selector '{selector}': {selector_error!s}"
            else:
                # Extract text from the entire page
                result = await page.content()

            # Release the browser
            await self._session_manager.release_async_browser(thread_id)

            return result
        except Exception as e:
            return f"Error extracting text: {e!s}"

    def extract_hyperlinks(
        self,
        thread_id: str = "default",
    ) -> str:
        """
        Extract hyperlinks from the current page (synchronous version).

        Args:
            thread_id (str): Thread ID for the browser session.

        Returns:
            str: The extracted hyperlinks.

        """
        try:
            # Get browser and extract hyperlinks
            browser = self._session_manager.get_sync_browser(thread_id)
            page = get_current_page(browser)

            # Extract all hyperlinks from the page
            links = page.eval_on_selector_all(
                "a[href]",
                """
                (elements) => {
                    return elements.map(el => {
                        return {
                            text: el.innerText || el.textContent,
                            href: el.href
                        };
                    });
                }
            """,
            )

            # Format the links
            formatted_links = []
            for i, link in enumerate(links):
                formatted_links.append(
                    f"{i + 1}. {link.get('text', 'No text')}: {link.get('href', 'No href')}"
                )

            result = (
                "\n".join(formatted_links)
                if formatted_links
                else "No hyperlinks found on the page"
            )

            # Release the browser
            self._session_manager.release_sync_browser(thread_id)

            return result
        except Exception as e:
            return f"Error extracting hyperlinks: {e!s}"

    async def aextract_hyperlinks(
        self,
        thread_id: str = "default",
    ) -> str:
        """
        Extract hyperlinks from the current page (asynchronous version).

        Args:
            thread_id (str): Thread ID for the browser session.

        Returns:
            str: The extracted hyperlinks.

        """
        try:
            # Get browser and extract hyperlinks
            browser = await self._session_manager.get_async_browser(thread_id)
            page = await aget_current_page(browser)

            # Extract all hyperlinks from the page
            links = await page.eval_on_selector_all(
                "a[href]",
                """
                (elements) => {
                    return elements.map(el => {
                        return {
                            text: el.innerText || el.textContent,
                            href: el.href
                        };
                    });
                }
            """,
            )

            # Format the links
            formatted_links = []
            for i, link in enumerate(links):
                formatted_links.append(
                    f"{i + 1}. {link.get('text', 'No text')}: {link.get('href', 'No href')}"
                )

            result = (
                "\n".join(formatted_links)
                if formatted_links
                else "No hyperlinks found on the page"
            )

            # Release the browser
            await self._session_manager.release_async_browser(thread_id)

            return result
        except Exception as e:
            return f"Error extracting hyperlinks: {e!s}"

    def get_elements(
        self,
        selector: str,
        thread_id: str = "default",
    ) -> str:
        """
        Get elements matching a CSS selector (synchronous version).

        Args:
            selector (str): CSS selector for the elements to get.
            thread_id (str): Thread ID for the browser session.

        Returns:
            str: Information about the matching elements.

        """
        try:
            # Get browser and find elements
            browser = self._session_manager.get_sync_browser(thread_id)
            page = get_current_page(browser)

            # Find elements matching the selector
            elements = page.query_selector_all(selector)

            if not elements:
                result = f"No elements found matching selector '{selector}'"
            else:
                # Extract information about the elements
                elements_info = []
                for i, element in enumerate(elements):
                    tag_name = element.evaluate("el => el.tagName.toLowerCase()")
                    text = element.text_content() or ""
                    attributes = element.evaluate("""
                        (el) => {
                            const attrs = {};
                            for (const attr of el.attributes) {
                                attrs[attr.name] = attr.value;
                            }
                            return attrs;
                        }
                    """)

                    # Format element info
                    attr_str = ", ".join([f'{k}="{v}"' for k, v in attributes.items()])
                    elements_info.append(
                        f"{i + 1}. <{tag_name} {attr_str}>{text}</{tag_name}>"
                    )

                result = (
                    f"Found {len(elements)} element(s) matching selector '{selector}':\n"
                    + "\n".join(elements_info)
                )

            # Release the browser
            self._session_manager.release_sync_browser(thread_id)

            return result
        except Exception as e:
            return f"Error getting elements: {e!s}"

    async def aget_elements(
        self,
        selector: str,
        thread_id: str = "default",
    ) -> str:
        """
        Get elements matching a CSS selector (asynchronous version).

        Args:
            selector (str): CSS selector for the elements to get.
            thread_id (str): Thread ID for the browser session.

        Returns:
            str: Information about the matching elements.

        """
        try:
            # Get browser and find elements
            browser = await self._session_manager.get_async_browser(thread_id)
            page = await aget_current_page(browser)

            # Find elements matching the selector
            elements = await page.query_selector_all(selector)

            if not elements:
                result = f"No elements found matching selector '{selector}'"
            else:
                # Extract information about the elements
                elements_info = []
                for i, element in enumerate(elements):
                    tag_name = await element.evaluate("el => el.tagName.toLowerCase()")
                    text = await element.text_content() or ""
                    attributes = await element.evaluate("""
                        (el) => {
                            const attrs = {};
                            for (const attr of el.attributes) {
                                attrs[attr.name] = attr.value;
                            }
                            return attrs;
                        }
                    """)

                    # Format element info
                    attr_str = ", ".join([f'{k}="{v}"' for k, v in attributes.items()])
                    elements_info.append(
                        f"{i + 1}. <{tag_name} {attr_str}>{text}</{tag_name}>"
                    )

                result = (
                    f"Found {len(elements)} element(s) matching selector '{selector}':\n"
                    + "\n".join(elements_info)
                )

            # Release the browser
            await self._session_manager.release_async_browser(thread_id)

            return result
        except Exception as e:
            return f"Error getting elements: {e!s}"

    def navigate_back(
        self,
        thread_id: str = "default",
    ) -> str:
        """
        Navigate to the previous page (synchronous version).

        Args:
            thread_id (str): Thread ID for the browser session.

        Returns:
            str: Confirmation message.

        """
        try:
            # Get browser and navigate back
            browser = self._session_manager.get_sync_browser(thread_id)
            page = get_current_page(browser)

            # Navigate back
            response = page.go_back()

            # Get the current URL after navigating back
            current_url = page.url if response else "unknown"

            # Release the browser
            self._session_manager.release_sync_browser(thread_id)

            if response:
                return f"Navigated back to {current_url}"
            else:
                return "Could not navigate back (no previous page in history)"
        except Exception as e:
            return f"Error navigating back: {e!s}"

    async def anavigate_back(
        self,
        thread_id: str = "default",
    ) -> str:
        """
        Navigate to the previous page (asynchronous version).

        Args:
            thread_id (str): Thread ID for the browser session.

        Returns:
            str: Confirmation message.

        """
        try:
            # Get browser and navigate back
            browser = await self._session_manager.get_async_browser(thread_id)
            page = await aget_current_page(browser)

            # Navigate back
            response = await page.go_back()

            # Get the current URL after navigating back
            current_url = page.url if response else "unknown"

            # Release the browser
            await self._session_manager.release_async_browser(thread_id)

            if response:
                return f"Navigated back to {current_url}"
            else:
                return "Could not navigate back (no previous page in history)"
        except Exception as e:
            return f"Error navigating back: {e!s}"

    def current_webpage(
        self,
        thread_id: str = "default",
    ) -> str:
        """
        Get information about the current webpage (synchronous version).

        Args:
            thread_id (str): Thread ID for the browser session.

        Returns:
            str: Information about the current webpage.

        """
        try:
            # Get browser and get current webpage info
            browser = self._session_manager.get_sync_browser(thread_id)
            page = get_current_page(browser)

            # Get the current URL
            url = page.url

            # Get the page title
            title = page.title()

            # Get basic page metrics
            metrics = page.evaluate("""
                () => {
                    return {
                        width: document.documentElement.clientWidth,
                        height: document.documentElement.clientHeight,
                        links: document.querySelectorAll('a').length,
                        images: document.querySelectorAll('img').length,
                        forms: document.querySelectorAll('form').length
                    }
                }
            """)

            # Format the result
            result = f"Current webpage information:\n"
            result += f"URL: {url}\n"
            result += f"Title: {title}\n"
            result += f"Viewport size: {metrics['width']}x{metrics['height']}\n"
            result += f"Links: {metrics['links']}\n"
            result += f"Images: {metrics['images']}\n"
            result += f"Forms: {metrics['forms']}"

            # Release the browser
            self._session_manager.release_sync_browser(thread_id)

            return result
        except Exception as e:
            return f"Error getting current webpage information: {e!s}"

    async def acurrent_webpage(
        self,
        thread_id: str = "default",
    ) -> str:
        """
        Get information about the current webpage (asynchronous version).

        Args:
            thread_id (str): Thread ID for the browser session.

        Returns:
            str: Information about the current webpage.

        """
        try:
            # Get browser and get current webpage info
            browser = await self._session_manager.get_async_browser(thread_id)
            page = await aget_current_page(browser)

            # Get the current URL
            url = page.url

            # Get the page title
            title = await page.title()

            # Get basic page metrics
            metrics = await page.evaluate("""
                () => {
                    return {
                        width: document.documentElement.clientWidth,
                        height: document.documentElement.clientHeight,
                        links: document.querySelectorAll('a').length,
                        images: document.querySelectorAll('img').length,
                        forms: document.querySelectorAll('form').length
                    }
                }
            """)

            # Format the result
            result = f"Current webpage information:\n"
            result += f"URL: {url}\n"
            result += f"Title: {title}\n"
            result += f"Viewport size: {metrics['width']}x{metrics['height']}\n"
            result += f"Links: {metrics['links']}\n"
            result += f"Images: {metrics['images']}\n"
            result += f"Forms: {metrics['forms']}"

            # Release the browser
            await self._session_manager.release_async_browser(thread_id)

            return result
        except Exception as e:
            return f"Error getting current webpage information: {e!s}"

    async def cleanup(self, thread_id: Optional[str] = None) -> None:
        """
        Clean up resources

        Args:
            thread_id: Optional thread ID to clean up. If None, cleans up all sessions.

        """
        if thread_id:
            # Clean up a specific thread's session
            if thread_id in self._browser_clients:
                try:
                    self._browser_clients[thread_id].stop()
                    del self._browser_clients[thread_id]
                    logger.info(f"Browser session for thread {thread_id} cleaned up")
                except Exception as e:
                    logger.warning(
                        f"Error stopping browser for thread {thread_id}: {e}"
                    )
        else:
            # Clean up all sessions
            thread_ids = list(self._browser_clients.keys())
            for tid in thread_ids:
                try:
                    self._browser_clients[tid].stop()
                except Exception as e:
                    logger.warning(f"Error stopping browser for thread {tid}: {e}")

            self._browser_clients = {}
            logger.info("All browser sessions cleaned up")
