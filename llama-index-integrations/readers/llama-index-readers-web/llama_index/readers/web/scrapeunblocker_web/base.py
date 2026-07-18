"""ScrapeUnblocker Web Reader."""

import json
import time
from typing import Any, Dict, List, Optional, Union

import requests
from llama_index.core.bridge.pydantic import Field, PrivateAttr, field_validator
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document


class ScrapeUnblockerWebReader(BasePydanticReader):
    """
    ScrapeUnblocker Web Reader.

    Read web pages through the ScrapeUnblocker API, which renders pages in a real
    browser behind anti-bot protections (Cloudflare, DataDome, PerimeterX, Akamai)
    and returns the resulting HTML or AI-parsed JSON.

    Args:
        api_key (str): ScrapeUnblocker API key. Get one at https://www.scrapeunblocker.com
        parsed_data (Optional[bool]): Return AI-parsed structured JSON instead of raw
            HTML. Default False.
        proxy_country (Optional[str]): Two-letter country code for the exit IP, for
            geo-restricted or localised content (e.g. "us", "de").
        time_sleep (Optional[int]): Seconds to wait after page load before capturing,
            for content that streams in late.
        get_cookies (Optional[bool]): Also return the cookies set by the target site.
            Default False.
        method (Optional[str]): Page interaction to perform before capturing, such as
            waiting for a selector.
        value (Optional[str]): Argument for `method` (for example the CSS selector to
            wait for).
        method_timeout (Optional[int]): Seconds to wait for `method` before giving up.
        base_url (Optional[str]): API base URL. Override to target staging.
        timeout (Optional[int]): Client-side HTTP timeout in seconds. Default 180.

    Examples:
        `pip install llama-index-readers-web`

        ```python
        from llama_index.readers.web import ScrapeUnblockerWebReader

        reader = ScrapeUnblockerWebReader(api_key="your-api-key")
        documents = reader.load_data(urls=["https://example.com"])
        ```

    """

    is_remote: bool = True
    api_key: str = Field(description="ScrapeUnblocker API key")
    parsed_data: Optional[bool] = Field(
        default=False,
        description="Return AI-parsed structured JSON instead of raw HTML.",
    )
    proxy_country: Optional[str] = Field(
        default=None,
        description="Two-letter country code for the exit IP, for geo-restricted content.",
    )
    time_sleep: Optional[int] = Field(
        default=None,
        description="Seconds to wait after page load before capturing the page.",
    )
    get_cookies: Optional[bool] = Field(
        default=False,
        description="Also return the cookies set by the target site.",
    )
    method: Optional[str] = Field(
        default=None,
        description="Page interaction to perform before capturing, e.g. waiting for a selector.",
    )
    value: Optional[str] = Field(
        default=None,
        description="Argument for `method`, e.g. the CSS selector to wait for.",
    )
    method_timeout: Optional[int] = Field(
        default=None,
        description="Seconds to wait for `method` before giving up.",
    )
    base_url: Optional[str] = Field(
        default="https://api.scrapeunblocker.com",
        description="API base URL. Override to target staging.",
    )
    timeout: Optional[int] = Field(
        default=180,
        description="Client-side HTTP timeout in seconds.",
    )

    _endpoint: str = PrivateAttr(default="/getPageSource")

    @field_validator("proxy_country")
    @classmethod
    def validate_proxy_country(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if not isinstance(v, str) or len(v) != 2 or not v.isalpha():
            raise ValueError(
                "proxy_country must be a two-letter country code, e.g. 'us' or 'de'"
            )
        return v.lower()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not self.api_key:
            raise ValueError("api_key is required")

    @classmethod
    def class_name(cls) -> str:
        return "ScrapeUnblockerWebReader"

    def _prepare_request_params(
        self, url: str, extra_params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Build the query string for one request."""
        params: Dict[str, Any] = {
            "url": url,
            "parsed_data": self.parsed_data,
            "get_cookies": self.get_cookies,
            "proxy_country": self.proxy_country,
            "time_sleep": self.time_sleep,
            "method": self.method,
            "value": self.value,
            "method_timeout": self.method_timeout,
        }

        if extra_params:
            params.update(extra_params)

        # The API treats an absent parameter as "use the default", so drop empties
        # rather than sending nulls.
        return {k: v for k, v in params.items() if v is not None and v is not False}

    def _make_request(
        self, url: str, extra_params: Optional[Dict] = None
    ) -> requests.Response:
        """Call the ScrapeUnblocker API for a single URL."""
        response = requests.post(
            f"{self.base_url.rstrip('/')}{self._endpoint}",
            params=self._prepare_request_params(url, extra_params),
            headers={"X-ScrapeUnblocker-Key": self.api_key},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response

    def _extract_metadata(
        self, response: requests.Response, url: str
    ) -> Dict[str, Any]:
        """Collect response metadata for the Document."""
        return {
            "source_url": url,
            "scraped_at": time.time(),
            "status_code": response.status_code,
            "content_type": response.headers.get("Content-Type", ""),
            "content_length": len(response.content),
            "scrapeunblocker_config": {
                "parsed_data": self.parsed_data,
                "proxy_country": self.proxy_country,
                "get_cookies": self.get_cookies,
                "method": self.method,
            },
        }

    def _process_response_content(self, response: requests.Response) -> str:
        """Return page text; parsed_data responses come back as JSON."""
        if self.parsed_data:
            try:
                return json.dumps(response.json(), ensure_ascii=False)
            except ValueError:
                # Fall back to the raw body if the API returned HTML anyway.
                return response.text
        return response.text

    def load_data(
        self, urls: Union[str, List[str]], extra_params: Optional[Dict] = None, **kwargs
    ) -> List[Document]:
        """
        Load data from URLs using the ScrapeUnblocker API.

        Args:
            urls: Single URL string or list of URLs to scrape
            extra_params: Additional query parameters for this specific request
            **kwargs: Additional keyword arguments (for compatibility)

        Returns:
            List of Document objects containing scraped content and metadata

        """
        if isinstance(urls, str):
            urls = [urls]

        documents = []

        for url in urls:
            try:
                response = self._make_request(url, extra_params)
                documents.append(
                    Document(
                        text=self._process_response_content(response),
                        metadata=self._extract_metadata(response, url),
                    )
                )
            except Exception as e:
                # One unreachable URL should not discard the rest of the batch.
                documents.append(
                    Document(
                        text=f"Error scraping {url}: {e!s}",
                        metadata={
                            "source_url": url,
                            "error": str(e),
                            "scraped_at": time.time(),
                            "status": "failed",
                        },
                    )
                )

        return documents
