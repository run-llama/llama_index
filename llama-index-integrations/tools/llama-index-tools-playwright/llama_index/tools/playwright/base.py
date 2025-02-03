from typing import Optional, Any, List, Sequence, TYPE_CHECKING
import json
from urllib.parse import urlparse

if TYPE_CHECKING:
    from playwright.sync_api import Browser as SyncBrowser
    from playwright.sync_api import Page as SyncPage
else:
    try:
        from playwright.sync_api import Browser as SyncBrowser
        from playwright.sync_api import Page as SyncPage
    except ImportError:
        raise ImportError(
            "The 'playwright' package is required to use this tool."
            " Please install it with 'pip install playwright'."
        )

from llama_index.core.tools.tool_spec.base import BaseToolSpec


def lazy_import_playwright_browsers():
    """
    Lazy import playwright browsers.
    """


class PlaywrightToolSpec(BaseToolSpec):
    """
    Playwright tool spec.
    """

    spec_functions = [
        "click",
        "get_current_page",
        "extract_hyperlinks",
        "extract_text",
        "get_elements",
        "navigate_to",
        "navigate_back",
    ]

    def __init__(
        self,
        sync_browser: Optional[SyncBrowser] = None,
        visible_only: bool = True,
        playwright_strict: bool = False,
        playwright_timeout: float = 1_000,
        absolute_url: bool = False,
    ) -> None:
        """
        Initialize PlaywrightToolSpec.

        Args:
            sync_browser: Optional[SyncBrowser] = None. A browser instance to use for automation.
            visible_only: bool = True. Whether to only click on visible elements.
            playwright_strict: bool = False. Whether to use strict mode for playwright.
            playwright_timeout: float = 1_000. Timeout for playwright operations.
            absolute_url: bool = False. Whether to return absolute urls.

        """
        lazy_import_playwright_browsers()
        self.sync_browser = sync_browser

        # for click tool
        self.visible_only = visible_only
        self.playwright_strict = playwright_strict
        self.playwright_timeout = playwright_timeout

        # for extractHyperlinks tool
        self.absolute_url = absolute_url

    @classmethod
    def from_sync_browser(cls, sync_browser: SyncBrowser) -> "PlaywrightToolSpec":
        """
        Initialize PlaywrightToolSpec from a sync browser instance.
        """
        return cls(sync_browser=sync_browser)

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
    def create_sync_playwright_browser(
        headless: bool = True, args: Optional[List[str]] = None
    ) -> SyncBrowser:
        """
        Create a playwright browser.

        Args:
            headless: Whether to run the browser in headless mode. Defaults to True.
            args: arguments to pass to browser.chromium.launch

        Returns:
            SyncBrowser: The playwright browser.
        """
        from playwright.sync_api import sync_playwright

        browser = sync_playwright().start()
        return browser.chromium.launch(headless=headless, args=args)

    def _get_current_page(self, browser: SyncBrowser) -> SyncPage:
        """
        Get the current page of the browser.

        Args:
            browser: The browser to get the current page from.

        Returns:
            SyncPage: The current page.
        """
        if not browser.contexts:
            context = browser.new_context()
            return context.new_page()
        context = browser.contexts[
            0
        ]  # Assuming you're using the default browser context
        if not context.pages:
            return context.new_page()
        # Assuming the last page in the list is the active one
        return context.pages[-1]

    def _check_bs_import(self):
        """Check that the arguments are valid."""
        try:
            from bs4 import BeautifulSoup  # noqa: F401
        except ImportError:
            raise ImportError(
                "The 'beautifulsoup4' package is required to use this tool."
                " Please install it with 'pip install beautifulsoup4'."
            )

    #################
    # Click #
    #################
    def click(
        self,
        selector: str,
    ) -> None:
        """
        Click on a en element based on a CSS selector.

        Args:
            selector: The CSS selector to click on.
        """
        if self.sync_browser is None:
            raise ValueError("Sync browser is not initialized")

        page = self._get_current_page(self.sync_browser)
        # Navigate to the desired webpage before using this tool
        selector_effective = self._selector_effective(selector=selector)
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

        try:
            page.click(
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
    def fill(
        self,
        selector: str,
        value: str,
    ) -> None:
        """
        Fill an input field with the given value.

        Args:
            selector: The CSS selector to fill.
            value: The value to fill in.
        """
        if self.sync_browser is None:
            raise ValueError("Sync browser is not initialized")

        page = self._get_current_page(self.sync_browser)
        # Navigate to the desired webpage before using this tool
        selector_effective = self._selector_effective(selector=selector)
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

        try:
            page.fill(
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
    def get_current_page(self) -> str:
        """
        Get the url of the current page.
        """
        if self.sync_browser is None:
            raise ValueError("Sync browser is not initialized")
        page = self._get_current_page(self.sync_browser)
        return page.url

    #################
    # Extract Hyperlinks #
    #################
    @staticmethod
    def scrape_page(page: Any, html_content: str, absolute_urls: bool) -> str:
        from urllib.parse import urljoin

        from bs4 import BeautifulSoup

        # Parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(html_content, "lxml")

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

    def extract_hyperlinks(self) -> str:
        """
        Extract all hyperlinks from the current page.
        """
        if self.sync_browser is None:
            raise ValueError("Sync browser is not initialized")
        self._check_bs_import()

        page = self._get_current_page(self.sync_browser)
        html_content = page.content()
        return self.scrape_page(page, html_content, self.absolute_url)

    #################
    # Extract Text #
    #################
    def extract_text(self) -> str:
        """
        Extract all text from the current page.
        """
        if self.sync_browser is None:
            raise ValueError("Sync browser is not initialized")
        self._check_bs_import()

        from bs4 import BeautifulSoup

        page = self._get_current_page(self.sync_browser)
        html_content = page.content()

        # Parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(html_content, "lxml")

        return " ".join(text for text in soup.stripped_strings)

    #################
    # Get Elements #
    #################
    def _get_elements(
        self, page: SyncPage, selector: str, attributes: Sequence[str]
    ) -> List[dict]:
        """Get elements matching the given CSS selector."""
        elements = page.query_selector_all(selector)
        results = []
        for element in elements:
            result = {}
            for attribute in attributes:
                if attribute == "innerText":
                    val: Optional[str] = element.inner_text()
                else:
                    val = element.get_attribute(attribute)
                if val is not None and val.strip() != "":
                    result[attribute] = val
            if result:
                results.append(result)
        return results

    def get_elements(self, selector: str, attributes: List[str] = ["innerText"]) -> str:
        """
        Retrieve elements in the current web page matching the given CSS selector.

        Args:
            selector: CSS selector, such as '*', 'div', 'p', 'a', #id, .classname
            attribute: Set of attributes to retrieve for each element
        """
        if self.sync_browser is None:
            raise ValueError("Sync browser is not initialized")

        page = self._get_current_page(self.sync_browser)
        results = self._get_elements(page, selector, attributes)
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

    def navigate_to(
        self,
        url: str,
    ) -> str:
        """
        Navigate to the given url.
        """
        if self.sync_browser is None:
            raise ValueError("Sync browser is not initialized")
        self.validate_url(url)

        page = self._get_current_page(self.sync_browser)
        response = page.goto(url)
        status = response.status if response else "unknown"
        return f"Navigating to {url} returned status code {status}"

    #################
    # Navigate Back #
    #################
    def navigate_back(self) -> str:
        """
        Navigate back to the previous page.
        """
        if self.sync_browser is None:
            raise ValueError("Sync browser is not initialized")
        page = self._get_current_page(self.sync_browser)
        response = page.go_back()

        if response:
            return (
                f"Navigated back to the previous page with URL '{response.url}'."
                f" Status code {response.status}"
            )
        else:
            return "Unable to navigate back; no previous page in the history"
