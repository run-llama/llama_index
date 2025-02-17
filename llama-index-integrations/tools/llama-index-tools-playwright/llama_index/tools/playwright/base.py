from typing import Optional, Any, List, Sequence
import json
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

from playwright.async_api import Browser as AsyncBrowser
from playwright.async_api import Page as AsyncPage

from llama_index.core.tools.tool_spec.base import BaseToolSpec


class PlaywrightToolSpec(BaseToolSpec):
    """
    Playwright tool spec.
    """

    spec_functions = [
        "click",
        "fill",
        "get_current_page",
        "extract_hyperlinks",
        "extract_text",
        "get_elements",
        "navigate_to",
        "navigate_back",
    ]

    def __init__(
        self,
        async_browser: Optional[AsyncBrowser] = None,
        visible_only: bool = False,
        playwright_strict: bool = False,
        playwright_timeout: float = 1_000,
        absolute_url: bool = False,
        html_parser: str = "html.parser",
    ) -> None:
        """
        Initialize PlaywrightToolSpec.

        Args:
            async_browser: Optional[AsyncBrowser] = None. A browser instance to use for automation.
            visible_only: bool = True. Whether to only click on visible elements.
            playwright_strict: bool = False. Whether to use strict mode for playwright.
            playwright_timeout: float = 1_000. Timeout for playwright operations.
            absolute_url: bool = False. Whether to return absolute urls.
            html_parser: str = "html.parser". The html parser to use with BeautifulSoup

        """
        self.async_browser = async_browser

        # for click tool
        self.visible_only = visible_only
        self.playwright_strict = playwright_strict
        self.playwright_timeout = playwright_timeout

        # for extractHyperlinks tool
        self.absolute_url = absolute_url
        self.html_parser = html_parser

    @classmethod
    def from_async_browser(cls, async_browser: AsyncBrowser) -> "PlaywrightToolSpec":
        """
        Initialize PlaywrightToolSpec from an async browser instance.
        """
        return cls(async_browser=async_browser)

    #################
    # Utils Methods #
    #################
    def _selector_effective(self, selector: str) -> str:
        """
        Get the effective selector.
        """
        if not self.visible_only:
            return selector
        return f"{selector} >> visible=1"

    @staticmethod
    async def create_async_playwright_browser(
        headless: bool = True, args: Optional[List[str]] = None
    ) -> AsyncBrowser:
        """
        Create an async playwright browser.

        Args:
            headless: Whether to run the browser in headless mode. Defaults to True.
            args: arguments to pass to browser.chromium.launch

        Returns:
            AsyncBrowser: The playwright browser.
        """
        from playwright.async_api import async_playwright

        browser = await async_playwright().start()
        return await browser.chromium.launch(headless=headless, args=args)

    async def _aget_current_page(self, browser: AsyncBrowser) -> AsyncPage:
        """
        Get the current page of the async browser.

        Args:
            browser: The browser to get the current page from.

        Returns:
            AsyncPage: The current page.
        """
        if not browser.contexts:
            context = await browser.new_context()
            return await context.new_page()
        context = browser.contexts[
            0
        ]  # Assuming you're using the default browser context
        if not context.pages:
            return await context.new_page()
        # Assuming the last page in the list is the active one
        return context.pages[-1]

    #################
    # Click #
    #################
    async def click(
        self,
        selector: str,
    ) -> str:
        """
        Click on a web element based on a CSS selector.

        Args:
            selector: The CSS selector for the web element to click on.
        """
        if self.async_browser is None:
            raise ValueError("Async browser is not initialized")

        page = await self._aget_current_page(self.async_browser)
        # Navigate to the desired webpage before using this tool
        selector_effective = self._selector_effective(selector=selector)
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError

        try:
            await page.click(
                selector_effective,
                strict=self.playwright_strict,
                timeout=self.playwright_timeout,
            )
        except PlaywrightTimeoutError:
            return f"Unable to click on element '{selector}'"
        return f"Clicked element '{selector}'"

    #################
    # Fill #
    #################
    async def fill(
        self,
        selector: str,
        value: str,
    ) -> str:
        """
        Fill an web input field specified by the given CSS selector with the given value.

        Args:
            selector: The CSS selector for the web input field to fill.
            value: The value to fill in.
        """
        if self.async_browser is None:
            raise ValueError("Async browser is not initialized")

        page = await self._aget_current_page(self.async_browser)
        # Navigate to the desired webpage before using this tool
        selector_effective = self._selector_effective(selector=selector)
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError

        try:
            await page.fill(
                selector_effective,
                value,
                strict=self.playwright_strict,
                timeout=self.playwright_timeout,
            )
        except PlaywrightTimeoutError:
            return f"Unable to fill element '{selector}'"
        return f"Filled element '{selector}'"

    #################
    # Get Current Page #
    #################
    async def get_current_page(self) -> str:
        """
        Get the url of the current web page.
        """
        if self.async_browser is None:
            raise ValueError("Async browser is not initialized")
        page = await self._aget_current_page(self.async_browser)
        return page.url

    #################
    # Extract Hyperlinks #
    #################
    def scrape_page(self, page: Any, html_content: str, absolute_urls: bool) -> str:
        # Parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(html_content, self.html_parser)

        # Find all the anchor elements and extract their href attributes
        anchors = soup.find_all("a")
        if absolute_urls:
            base_url = page.url
            links = [urljoin(base_url, anchor.get("href", "")) for anchor in anchors]
        else:
            links = [anchor.get("href", "") for anchor in anchors]
        # Return the list of links as a JSON string. Duplicated link
        # only appears once in the list
        return json.dumps(list(set(links)))

    async def extract_hyperlinks(self) -> str:
        """
        Extract all hyperlinks from the current web page.
        """
        if self.async_browser is None:
            raise ValueError("Async browser is not initialized")

        page = await self._aget_current_page(self.async_browser)
        html_content = await page.content()
        return self.scrape_page(page, html_content, self.absolute_url)

    #################
    # Extract Text #
    #################
    async def extract_text(self) -> str:
        """
        Extract all text from the current web page.
        """
        if self.async_browser is None:
            raise ValueError("Async browser is not initialized")

        page = await self._aget_current_page(self.async_browser)
        html_content = await page.content()

        # Parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(html_content, self.html_parser)

        return " ".join(text for text in soup.stripped_strings)

    #################
    # Get Elements #
    #################
    async def _aget_elements(
        self, page: AsyncPage, selector: str, attributes: Sequence[str]
    ) -> List[dict]:
        """Get elements matching the given CSS selector."""
        elements = await page.query_selector_all(selector)
        results = []
        for element in elements:
            result = {}
            for attribute in attributes:
                if attribute == "innerText":
                    val: Optional[str] = await element.inner_text()
                else:
                    val = await element.get_attribute(attribute)
                if val is not None and val.strip() != "":
                    result[attribute] = val
            if result:
                results.append(result)
        return results

    async def get_elements(
        self, selector: str, attributes: List[str] = ["innerText"]
    ) -> str:
        """
        Retrieve elements in the current web page matching the given CSS selector.

        Args:
            selector: CSS selector, such as '*', 'div', 'p', 'a', #id, .classname
            attribute: Set of attributes to retrieve for each element
        """
        if self.async_browser is None:
            raise ValueError("Async browser is not initialized")

        page = await self._aget_current_page(self.async_browser)
        results = await self._aget_elements(page, selector, attributes)
        return json.dumps(results, ensure_ascii=False)

    #################
    # Navigate #
    #################
    def validate_url(self, url: str):
        """
        Validate the given url.
        """
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError("URL scheme must be 'http' or 'https'")

    async def navigate_to(
        self,
        url: str,
    ) -> str:
        """
        Navigate to the given url.

        Args:
            url: The url to navigate to.
        """
        if self.async_browser is None:
            raise ValueError("Async browser is not initialized")
        self.validate_url(url)

        page = await self._aget_current_page(self.async_browser)
        response = await page.goto(url)
        status = response.status if response else "unknown"
        return f"Navigating to {url} returned status code {status}"

    #################
    # Navigate Back #
    #################
    async def navigate_back(self) -> str:
        """
        Navigate back to the previous web page.
        """
        if self.async_browser is None:
            raise ValueError("Async browser is not initialized")
        page = await self._aget_current_page(self.async_browser)
        response = await page.go_back()

        if response:
            return (
                f"Navigated back to the previous page with URL '{response.url}'."
                f" Status code {response.status}"
            )
        else:
            return "Unable to navigate back; no previous page in the history"
