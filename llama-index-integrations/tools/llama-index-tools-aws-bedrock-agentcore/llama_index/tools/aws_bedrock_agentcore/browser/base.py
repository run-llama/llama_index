import asyncio
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from llama_index.core.tools.tool_spec.base import BaseToolSpec

from bedrock_agentcore.tools.browser_client import BrowserClient

from llama_index.tools.aws_bedrock_agentcore.utils import get_aws_region
from .browser_session_manager import BrowserSessionManager
from .utils import aget_current_page, get_current_page

DEFAULT_BROWSER_IDENTIFIER = "aws.browser.v1"
DEFAULT_BROWSER_SESSION_TIMEOUT = 3600
DEFAULT_BROWSER_LIVE_VIEW_PRESIGNED_URL_TIMEOUT = 300

logger = logging.getLogger(__name__)


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
        ("generate_live_view_url", "agenerate_live_view_url"),
        ("list_browsers", "alist_browsers"),
        ("create_browser", "acreate_browser"),
        ("delete_browser", "adelete_browser"),
        ("get_browser", "aget_browser"),
        ("take_control", "atake_control"),
        ("release_control", "arelease_control"),
    ]

    def __init__(
        self,
        region: Optional[str] = None,
        identifier: Optional[str] = None,
    ) -> None:
        """
        Initialize the AWS Bedrock AgentCore Browser Tool Spec.

        Args:
            region (Optional[str]): AWS region to use for Bedrock AgentCore services.
                If not provided, will try to get it from environment variables.
            identifier (Optional[str]): Custom browser identifier for VPC-enabled
                resources. If not provided, uses the default identifier.

        """
        self.region = region if region is not None else get_aws_region()
        self._identifier = identifier
        self._browser_clients: Dict[str, BrowserClient] = {}
        self._cp_browser_client: Optional[BrowserClient] = None
        self._session_manager = BrowserSessionManager(
            region=self.region, identifier=self._identifier
        )

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
        browser_client = BrowserClient(self.region, integration_source="llamaindex")
        self._browser_clients[thread_id] = browser_client
        return browser_client

    def _get_control_plane_client(self) -> BrowserClient:
        """
        Get or create a browser client for control-plane operations only.

        This client is used for account-level operations (list, create, delete, get)
        that do not require a browser session.
        """
        if self._cp_browser_client is None:
            self._cp_browser_client = BrowserClient(
                self.region, integration_source="llamaindex"
            )
        return self._cp_browser_client

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
            try:
                page = get_current_page(browser)
                response = page.goto(url)
                status = response.status if response else "unknown"
                return f"Navigated to {url} with status code {status}"
            finally:
                self._session_manager.release_sync_browser(thread_id)
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
            try:
                page = get_current_page(browser)

                try:
                    page.click(selector, timeout=5000)
                    return f"Clicked on element with selector '{selector}'"
                except Exception as click_error:
                    return f"Unable to click on element with selector '{selector}': {click_error!s}"
            finally:
                self._session_manager.release_sync_browser(thread_id)
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
            try:
                page = get_current_page(browser)

                if selector:
                    try:
                        element = page.query_selector(selector)
                        if element:
                            text = element.text_content()
                            result = (
                                text if text else "Element found but contains no text"
                            )
                        else:
                            result = f"No element found with selector '{selector}'"
                    except Exception as selector_error:
                        result = f"Error extracting text from selector '{selector}': {selector_error!s}"
                else:
                    # Extract text from the entire page
                    result = page.content()

                return result
            finally:
                self._session_manager.release_sync_browser(thread_id)
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
            try:
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

                return (
                    "\n".join(formatted_links)
                    if formatted_links
                    else "No hyperlinks found on the page"
                )
            finally:
                self._session_manager.release_sync_browser(thread_id)
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
            try:
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
                        attributes = element.evaluate(
                            """
                            (el) => {
                                const attrs = {};
                                for (const attr of el.attributes) {
                                    attrs[attr.name] = attr.value;
                                }
                                return attrs;
                            }
                        """
                        )

                        # Format element info
                        attr_str = ", ".join(
                            [f'{k}="{v}"' for k, v in attributes.items()]
                        )
                        elements_info.append(
                            f"{i + 1}. <{tag_name} {attr_str}>{text}</{tag_name}>"
                        )

                    result = (
                        f"Found {len(elements)} element(s) matching selector '{selector}':\n"
                        + "\n".join(elements_info)
                    )

                return result
            finally:
                self._session_manager.release_sync_browser(thread_id)
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
                    attributes = await element.evaluate(
                        """
                        (el) => {
                            const attrs = {};
                            for (const attr of el.attributes) {
                                attrs[attr.name] = attr.value;
                            }
                            return attrs;
                        }
                    """
                    )

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
            try:
                page = get_current_page(browser)

                # Navigate back
                response = page.go_back()

                # Get the current URL after navigating back
                current_url = page.url if response else "unknown"

                if response:
                    return f"Navigated back to {current_url}"
                else:
                    return "Could not navigate back (no previous page in history)"
            finally:
                self._session_manager.release_sync_browser(thread_id)
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
            try:
                page = get_current_page(browser)

                # Get the current URL
                url = page.url

                # Get the page title
                title = page.title()

                # Get basic page metrics
                metrics = page.evaluate(
                    """
                    () => {
                        return {
                            width: document.documentElement.clientWidth,
                            height: document.documentElement.clientHeight,
                            links: document.querySelectorAll('a').length,
                            images: document.querySelectorAll('img').length,
                            forms: document.querySelectorAll('form').length
                        }
                    }
                """
                )

                # Format the result
                result = f"Current webpage information:\n"
                result += f"URL: {url}\n"
                result += f"Title: {title}\n"
                result += f"Viewport size: {metrics['width']}x{metrics['height']}\n"
                result += f"Links: {metrics['links']}\n"
                result += f"Images: {metrics['images']}\n"
                result += f"Forms: {metrics['forms']}"

                return result
            finally:
                self._session_manager.release_sync_browser(thread_id)
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
            metrics = await page.evaluate(
                """
                () => {
                    return {
                        width: document.documentElement.clientWidth,
                        height: document.documentElement.clientHeight,
                        links: document.querySelectorAll('a').length,
                        images: document.querySelectorAll('img').length,
                        forms: document.querySelectorAll('form').length
                    }
                }
            """
            )

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

    def generate_live_view_url(
        self,
        expires: int = DEFAULT_BROWSER_LIVE_VIEW_PRESIGNED_URL_TIMEOUT,
        thread_id: str = "default",
    ) -> str:
        """
        Generate a presigned URL for live viewing a browser session (synchronous version).

        This URL allows a human to observe the browser session in real-time for oversight.
        A browser session must already exist for the given thread_id (e.g., by navigating
        to a URL first).

        Args:
            expires (int): Seconds until the URL expires. Maximum 300. Default is 300.
            thread_id (str): Thread ID for the browser session. Default is "default".

        Returns:
            str: The presigned URL for viewing the browser session.

        """
        try:
            browser_client = self._session_manager.get_browser_client(thread_id)
            if browser_client is None:
                return (
                    f"No browser session found for thread '{thread_id}'. "
                    "Navigate to a URL first to start a session."
                )
            return browser_client.generate_live_view_url(expires=expires)
        except Exception as e:
            return f"Error generating live view URL: {e!s}"

    async def agenerate_live_view_url(
        self,
        expires: int = DEFAULT_BROWSER_LIVE_VIEW_PRESIGNED_URL_TIMEOUT,
        thread_id: str = "default",
    ) -> str:
        """
        Generate a presigned URL for live viewing a browser session (asynchronous version).

        This URL allows a human to observe the browser session in real-time for oversight.
        A browser session must already exist for the given thread_id (e.g., by navigating
        to a URL first).

        Args:
            expires (int): Seconds until the URL expires. Maximum 300. Default is 300.
            thread_id (str): Thread ID for the browser session. Default is "default".

        Returns:
            str: The presigned URL for viewing the browser session.

        """
        return await asyncio.to_thread(
            self.generate_live_view_url, expires=expires, thread_id=thread_id
        )

    def list_browsers(
        self,
        browser_type: Optional[str] = None,
        max_results: int = 10,
        thread_id: str = "default",
    ) -> str:
        """
        List all browsers in the account (synchronous version).

        Args:
            browser_type (Optional[str]): Filter by type: "SYSTEM" or "CUSTOM".
            max_results (int): Maximum results to return (1-100). Default is 10.
            thread_id (str): Deprecated. Ignored. Kept for backward compatibility.

        Returns:
            str: JSON-formatted list of browser summaries.

        """
        try:
            browser_client = self._get_control_plane_client()
            response = browser_client.list_browsers(
                browser_type=browser_type, max_results=max_results
            )
            summaries = response.get("browserSummaries", [])
            if not summaries:
                return "No browsers found."
            lines = []
            for b in summaries:
                lines.append(
                    f"- {b.get('name', 'N/A')} (ID: {b.get('browserId', 'N/A')}, "
                    f"Status: {b.get('status', 'N/A')}, Type: {b.get('type', 'N/A')})"
                )
            return f"Found {len(summaries)} browser(s):\n" + "\n".join(lines)
        except Exception as e:
            return f"Error listing browsers: {e!s}"

    async def alist_browsers(
        self,
        browser_type: Optional[str] = None,
        max_results: int = 10,
        thread_id: str = "default",
    ) -> str:
        """
        List all browsers in the account (asynchronous version).

        Args:
            browser_type (Optional[str]): Filter by type: "SYSTEM" or "CUSTOM".
            max_results (int): Maximum results to return (1-100). Default is 10.
            thread_id (str): Deprecated. Ignored. Kept for backward compatibility.

        Returns:
            str: JSON-formatted list of browser summaries.

        """
        return await asyncio.to_thread(
            self.list_browsers,
            browser_type=browser_type,
            max_results=max_results,
            thread_id=thread_id,
        )

    def create_browser(
        self,
        name: str,
        execution_role_arn: str,
        network_mode: str = "PUBLIC",
        description: str = "",
        subnet_ids: Optional[List[str]] = None,
        security_group_ids: Optional[List[str]] = None,
        thread_id: str = "default",
    ) -> str:
        """
        Create a custom browser with specific configuration (synchronous version).

        Args:
            name (str): Name for the browser. Must match pattern [a-zA-Z][a-zA-Z0-9_]{0,47}.
            execution_role_arn (str): IAM role ARN with permissions for browser operations.
            network_mode (str): Network mode: "PUBLIC" or "VPC". Default is "PUBLIC".
            description (str): Description of the browser. Default is "".
            subnet_ids (Optional[List[str]]): Subnet IDs for VPC mode.
            security_group_ids (Optional[List[str]]): Security group IDs for VPC mode.
            thread_id (str): Deprecated. Ignored. Kept for backward compatibility.

        Returns:
            str: Confirmation with browser ID and status.

        """
        try:
            browser_client = self._get_control_plane_client()
            network_config: Dict[str, Any] = {"networkMode": network_mode}
            if subnet_ids or security_group_ids:
                vpc_config: Dict[str, Any] = {}
                if subnet_ids:
                    vpc_config["subnets"] = subnet_ids
                if security_group_ids:
                    vpc_config["securityGroups"] = security_group_ids
                network_config["vpcConfig"] = vpc_config
            kwargs: Dict[str, Any] = {
                "name": name,
                "execution_role_arn": execution_role_arn,
                "network_configuration": network_config,
            }
            if description:
                kwargs["description"] = description
            response = browser_client.create_browser(**kwargs)
            browser_id = response.get("browserId", "unknown")
            status = response.get("status", "unknown")
            return f"Browser created (ID: {browser_id}, Status: {status})"
        except Exception as e:
            return f"Error creating browser: {e!s}"

    async def acreate_browser(
        self,
        name: str,
        execution_role_arn: str,
        network_mode: str = "PUBLIC",
        description: str = "",
        subnet_ids: Optional[List[str]] = None,
        security_group_ids: Optional[List[str]] = None,
        thread_id: str = "default",
    ) -> str:
        """
        Create a custom browser with specific configuration (asynchronous version).

        Args:
            name (str): Name for the browser. Must match pattern [a-zA-Z][a-zA-Z0-9_]{0,47}.
            execution_role_arn (str): IAM role ARN with permissions for browser operations.
            network_mode (str): Network mode: "PUBLIC" or "VPC". Default is "PUBLIC".
            description (str): Description of the browser. Default is "".
            subnet_ids (Optional[List[str]]): Subnet IDs for VPC mode.
            security_group_ids (Optional[List[str]]): Security group IDs for VPC mode.
            thread_id (str): Deprecated. Ignored. Kept for backward compatibility.

        Returns:
            str: Confirmation with browser ID and status.

        """
        return await asyncio.to_thread(
            self.create_browser,
            name=name,
            execution_role_arn=execution_role_arn,
            network_mode=network_mode,
            description=description,
            subnet_ids=subnet_ids,
            security_group_ids=security_group_ids,
            thread_id=thread_id,
        )

    def delete_browser(
        self,
        browser_id: str,
        thread_id: str = "default",
    ) -> str:
        """
        Delete a custom browser (synchronous version).

        Args:
            browser_id (str): The browser identifier to delete.
            thread_id (str): Deprecated. Ignored. Kept for backward compatibility.

        Returns:
            str: Confirmation of deletion.

        """
        try:
            browser_client = self._get_control_plane_client()
            response = browser_client.delete_browser(browser_id=browser_id)
            status = response.get("status", "unknown")
            return f"Browser '{browser_id}' deleted (Status: {status})"
        except Exception as e:
            return f"Error deleting browser: {e!s}"

    async def adelete_browser(
        self,
        browser_id: str,
        thread_id: str = "default",
    ) -> str:
        """
        Delete a custom browser (asynchronous version).

        Args:
            browser_id (str): The browser identifier to delete.
            thread_id (str): Deprecated. Ignored. Kept for backward compatibility.

        Returns:
            str: Confirmation of deletion.

        """
        return await asyncio.to_thread(
            self.delete_browser, browser_id=browser_id, thread_id=thread_id
        )

    def get_browser(
        self,
        browser_id: str,
        thread_id: str = "default",
    ) -> str:
        """
        Get detailed information about a browser (synchronous version).

        Args:
            browser_id (str): The browser identifier.
            thread_id (str): Deprecated. Ignored. Kept for backward compatibility.

        Returns:
            str: Browser details including name, status, and configuration.

        """
        try:
            browser_client = self._get_control_plane_client()
            response = browser_client.get_browser(browser_id=browser_id)
            name = response.get("name", "N/A")
            status = response.get("status", "N/A")
            desc = response.get("description", "")
            result = f"Browser '{browser_id}':\n"
            result += f"  Name: {name}\n"
            result += f"  Status: {status}\n"
            if desc:
                result += f"  Description: {desc}\n"
            network = response.get("networkConfiguration", {})
            if network:
                result += f"  Network mode: {network.get('networkMode', 'N/A')}"
            return result
        except Exception as e:
            return f"Error getting browser: {e!s}"

    async def aget_browser(
        self,
        browser_id: str,
        thread_id: str = "default",
    ) -> str:
        """
        Get detailed information about a browser (asynchronous version).

        Args:
            browser_id (str): The browser identifier.
            thread_id (str): Deprecated. Ignored. Kept for backward compatibility.

        Returns:
            str: Browser details including name, status, and configuration.

        """
        return await asyncio.to_thread(
            self.get_browser, browser_id=browser_id, thread_id=thread_id
        )

    def take_control(
        self,
        thread_id: str = "default",
    ) -> str:
        """
        Take manual control of a browser session by disabling the automation stream (synchronous version).

        This allows a human to interact with the browser via the live view URL while
        preventing the automation agent from making changes.

        Args:
            thread_id (str): Thread ID for the browser session. Default is "default".

        Returns:
            str: Confirmation message.

        """
        try:
            browser_client = self._session_manager.get_browser_client(thread_id)
            if browser_client is None:
                return (
                    f"No browser session found for thread '{thread_id}'. "
                    "Navigate to a URL first to start a session."
                )
            browser_client.take_control()
            return "Took manual control of the browser session. Automation stream disabled."
        except Exception as e:
            return f"Error taking control: {e!s}"

    async def atake_control(
        self,
        thread_id: str = "default",
    ) -> str:
        """
        Take manual control of a browser session by disabling the automation stream (asynchronous version).

        Args:
            thread_id (str): Thread ID for the browser session. Default is "default".

        Returns:
            str: Confirmation message.

        """
        return await asyncio.to_thread(self.take_control, thread_id=thread_id)

    def release_control(
        self,
        thread_id: str = "default",
    ) -> str:
        """
        Release manual control and re-enable the automation stream (synchronous version).

        This returns control to the automation agent after manual interaction.

        Args:
            thread_id (str): Thread ID for the browser session. Default is "default".

        Returns:
            str: Confirmation message.

        """
        try:
            browser_client = self._session_manager.get_browser_client(thread_id)
            if browser_client is None:
                return (
                    f"No browser session found for thread '{thread_id}'. "
                    "Navigate to a URL first to start a session."
                )
            browser_client.release_control()
            return "Released manual control. Automation stream re-enabled."
        except Exception as e:
            return f"Error releasing control: {e!s}"

    async def arelease_control(
        self,
        thread_id: str = "default",
    ) -> str:
        """
        Release manual control and re-enable the automation stream (asynchronous version).

        Args:
            thread_id (str): Thread ID for the browser session. Default is "default".

        Returns:
            str: Confirmation message.

        """
        return await asyncio.to_thread(self.release_control, thread_id=thread_id)

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
