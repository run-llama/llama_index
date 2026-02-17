"""
Massive Web Reader for LlamaIndex.

This module provides a web page reader that uses the Massive proxy network
with Playwright browser automation for reliable web scraping with
geotargeting capabilities.
"""

import logging
from typing import Iterator, List, Optional

from bs4 import BeautifulSoup
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document
from playwright.async_api import async_playwright
from playwright.sync_api import (
    Error as PlaywrightError,
    TimeoutError as PlaywrightTimeout,
    sync_playwright,
)
from pydantic import PrivateAttr, field_validator

from llama_index.readers.massive.country_config import COUNTRY_CONFIG

logger = logging.getLogger(__name__)

DEFAULT_PAGE_LOAD_TIMEOUT_MS = 30000
MASSIVE_PROXY_ADDRESS = "https://network.joinmassive.com:65535"


class MassiveWebReader(BasePydanticReader):
    """
    Web reader using Massive proxy network with Playwright.

    Reads pages by rendering them with Playwright using Massive proxy with
    comprehensive geotargeting (country, city, zipcode, ASN), device type
    targeting, sticky sessions, and optional raw HTML output.

    Features:
        - Country targeting with locale and timezone support
        - City and ZIP code geotargeting
        - ASN (Autonomous System Number) targeting
        - Device type targeting (mobile, common, tv)
        - Sticky sessions with customizable TTL
        - Raw HTML mode (bypasses BeautifulSoup cleaning)

    Examples:
        Basic country targeting:
        >>> reader = MassiveWebReader(
        ...     username="your_username",
        ...     password="your_password",
        ...     country="US"
        ... )
        >>> docs = reader.load_data(["https://example.com"])

        City targeting with mobile device:
        >>> reader = MassiveWebReader(
        ...     username="your_username",
        ...     password="your_password",
        ...     country="US",
        ...     city="New York",
        ...     device_type="mobile"
        ... )

        Sticky session with custom TTL:
        >>> reader = MassiveWebReader(
        ...     username="your_username",
        ...     password="your_password",
        ...     country="GB",
        ...     session="my-session-123",
        ...     ttl=30
        ... )

    """

    is_remote: bool = True

    # Authentication (required)
    username: str
    password: str

    # Geotargeting
    country: Optional[str] = None
    city: Optional[str] = None
    zipcode: Optional[str] = None
    asn: Optional[str] = None

    # Device and session
    device_type: Optional[str] = None
    session: Optional[str] = None
    ttl: int = 15

    # Browser settings
    headless: bool = True
    page_load_timeout: int = DEFAULT_PAGE_LOAD_TIMEOUT_MS
    additional_wait_ms: Optional[int] = None

    # Output settings
    raw_html: bool = False

    _proxy: dict = PrivateAttr()

    @field_validator("device_type")
    @classmethod
    def validate_device_type(cls, v: Optional[str]) -> Optional[str]:
        """Validate device_type is one of allowed values."""
        if v is not None and v not in ("mobile", "common", "tv"):
            raise ValueError(
                f"Invalid device_type: {v}. Must be 'mobile', 'common', or 'tv'"
            )
        return v

    def __init__(self, **data):
        """
        Initialize MassiveWebReader.

        Args:
            username: Massive proxy username.
            password: Massive proxy password.
            country: Two-letter ISO 3166-1 alpha-2 country code (e.g., 'US', 'DE').
            city: City name for geotargeting (e.g., 'New York', 'London').
            zipcode: ZIP/postal code (e.g., '10001', 'SW1').
            asn: ASN identifier for network targeting.
            device_type: Device type - 'mobile', 'common', or 'tv'.
            session: Session identifier for sticky sessions.
            ttl: Session TTL in minutes (default: 15).
            headless: Whether to run browser in headless mode (default: True).
            page_load_timeout: Maximum time in ms to wait for page load (default: 30000).
            additional_wait_ms: Extra wait time after networkidle for lazy-loaded content.
            raw_html: If True, return raw HTML without BeautifulSoup processing.

        """
        super().__init__(**data)

        proxy_username = self.username

        if self.country:
            proxy_username += f"-country-{self.country}"
        if self.city:
            proxy_username += f"-city-{self.city}"
        if self.zipcode:
            proxy_username += f"-zipcode-{self.zipcode}"
        if self.asn:
            proxy_username += f"-asn-{self.asn}"
        if self.device_type:
            proxy_username += f"-type-{self.device_type}"
        if self.session:
            proxy_username += f"-session-{self.session}"
            proxy_username += f"-sessionttl-{self.ttl}"

        self._proxy = {
            "server": MASSIVE_PROXY_ADDRESS,
            "username": proxy_username,
            "password": self.password,
        }

    @classmethod
    def class_name(cls) -> str:
        """Return class name for serialization."""
        return "MassiveWebReader"

    def _get_browser_config(self) -> tuple[dict, dict, str]:
        """Return browser launch args, context options, and init script."""
        country = self.country or "DEFAULT"
        config = COUNTRY_CONFIG.get(country, COUNTRY_CONFIG["DEFAULT"])

        launch_args = {
            "headless": self.headless,
            "proxy": self._proxy,
            "args": ["--disable-blink-features=AutomationControlled"],
        }

        context_options = {
            "viewport": {"width": 1280, "height": 720},
            "locale": config["locale"],
            "timezone_id": config["timezone"],
            "permissions": ["geolocation"],
        }

        init_script = """
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """

        return launch_args, context_options, init_script

    def load_data(self, urls: List[str]) -> List[Document]:
        """
        Load data from URLs.

        Args:
            urls: List of URLs to scrape.

        Returns:
            List of Document objects containing page content.

        """
        return list(self.lazy_load_data(urls))

    def lazy_load_data(self, urls: List[str]) -> Iterator[Document]:
        """
        Lazily load data from URLs.

        Generator that yields documents one at a time, allowing for
        memory-efficient processing of large URL lists.

        Args:
            urls: List of URLs to scrape.

        Yields:
            Document objects containing page content.

        """
        launch_args, context_options, init_script = self._get_browser_config()

        with sync_playwright() as p:
            browser = None
            try:
                logger.info(f"Launching browser (headless={self.headless})")
                browser = p.chromium.launch(**launch_args)
                context = browser.new_context(**context_options)
                context.add_init_script(init_script)
                page = context.new_page()

                for url in urls:
                    doc = self._load_single_url(page, url)
                    if doc:
                        yield doc

            finally:
                if browser:
                    try:
                        browser.close()
                        logger.debug("Browser closed successfully")
                    except Exception as e:
                        logger.error(f"Error closing browser: {e}")

    async def aload_data(self, urls: List[str]) -> List[Document]:
        """
        Asynchronously load data from URLs.

        Args:
            urls: List of URLs to scrape.

        Returns:
            List of Document objects containing page content.

        """
        launch_args, context_options, init_script = self._get_browser_config()
        documents = []

        async with async_playwright() as p:
            browser = None
            try:
                logger.info(f"Launching async browser (headless={self.headless})")
                browser = await p.chromium.launch(**launch_args)
                context = await browser.new_context(**context_options)
                await context.add_init_script(init_script)
                page = await context.new_page()

                for url in urls:
                    doc = await self._aload_single_url(page, url)
                    if doc:
                        documents.append(doc)

            finally:
                if browser:
                    try:
                        await browser.close()
                        logger.debug("Async browser closed successfully")
                    except Exception as e:
                        logger.error(f"Error closing async browser: {e}")

        return documents

    def _load_single_url(self, page, url: str) -> Optional[Document]:
        """Load a single URL synchronously."""
        try:
            logger.info(f"Navigating to {url}")
            page.goto(
                url,
                wait_until="networkidle",
                timeout=self.page_load_timeout,
            )

            if self.additional_wait_ms:
                logger.debug(f"Waiting additional {self.additional_wait_ms}ms")
                page.wait_for_timeout(self.additional_wait_ms)

            logger.debug(f"Page loaded successfully: {url}")
            content = page.content()

            return self._process_content(content, url)

        except PlaywrightTimeout as e:
            logger.warning(
                f"Timeout loading {url} after {self.page_load_timeout}ms: {e}"
            )
            return None
        except PlaywrightError as e:
            logger.error(f"Playwright error on {url}: {e}")
            return None
        except Exception as e:
            logger.critical(f"Unexpected error on {url}: {e}", exc_info=True)
            return None

    async def _aload_single_url(self, page, url: str) -> Optional[Document]:
        """Load a single URL asynchronously."""
        try:
            logger.info(f"Async navigating to {url}")
            await page.goto(
                url,
                wait_until="networkidle",
                timeout=self.page_load_timeout,
            )

            if self.additional_wait_ms:
                logger.debug(f"Async waiting additional {self.additional_wait_ms}ms")
                await page.wait_for_timeout(self.additional_wait_ms)

            logger.debug(f"Async page loaded successfully: {url}")
            content = await page.content()

            return self._process_content(content, url)

        except PlaywrightTimeout as e:
            logger.warning(
                f"Async timeout loading {url} after {self.page_load_timeout}ms: {e}"
            )
            return None
        except PlaywrightError as e:
            logger.error(f"Async Playwright error on {url}: {e}")
            return None
        except Exception as e:
            logger.critical(f"Async unexpected error on {url}: {e}", exc_info=True)
            return None

    def _process_content(self, content: str, url: str) -> Document:
        """Process HTML content into a Document."""
        if self.raw_html:
            logger.debug(f"Returned raw HTML for {url} ({len(content)} chars)")
            return Document(text=content, metadata={"url": url})

        # Clean up HTML with BeautifulSoup
        soup = BeautifulSoup(content, "html.parser")

        # Remove script and style elements
        for element in soup(["script", "style"]):
            element.extract()

        text = soup.get_text(separator="\n")

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)

        logger.debug(f"Processed {url} ({len(text)} chars)")
        return Document(text=text, metadata={"url": url})
