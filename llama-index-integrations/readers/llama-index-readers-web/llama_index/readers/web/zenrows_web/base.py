"""ZenRows Web Reader."""

import json
import time
from typing import Any, Dict, List, Literal, Optional, Union

import requests
from llama_index.core.bridge.pydantic import Field, PrivateAttr, field_validator
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document


class ZenRowsWebReader(BasePydanticReader):
    """
    ZenRows Web Reader.

    Read web pages using ZenRows Universal Scraper API with advanced features like:
    - JavaScript rendering for dynamic content
    - Anti-bot bypass
    - Premium residential proxies with geo-location
    - Custom headers and session management
    - Advanced data extraction with CSS selectors
    - Multiple output formats (HTML, Markdown, Text, PDF)
    - Screenshot capabilities

    Args:
        api_key (str): ZenRows API key. Get one at https://app.zenrows.com/register
        js_render (Optional[bool]): Enable JavaScript rendering with a headless browser. Default False.
        js_instructions (Optional[str]): Execute custom JavaScript on the page to interact with elements.
        premium_proxy (Optional[bool]): Use residential IPs to bypass anti-bot protection. Default False.
        proxy_country (Optional[str]): Set the country of the IP used for the request (requires Premium Proxies).
        session_id (Optional[int]): Maintain the same IP for multiple requests for up to 10 minutes.
        custom_headers (Optional[Dict[str, str]]): Include custom headers in your request to mimic browser behavior.
        wait_for (Optional[str]): Wait for a specific CSS Selector to appear in the DOM before returning content.
        wait (Optional[int]): Wait a fixed amount of milliseconds after page load.
        block_resources (Optional[str]): Block specific resources (images, fonts, etc.) from loading.
        response_type (Optional[Literal["markdown", "plaintext", "pdf"]]): Convert HTML to other formats.
        css_extractor (Optional[str]): Extract specific elements using CSS selectors (JSON format).
        autoparse (Optional[bool]): Automatically extract structured data from HTML. Default False.
        screenshot (Optional[str]): Capture an above-the-fold screenshot of the page.
        screenshot_fullpage (Optional[str]): Capture a full-page screenshot.
        screenshot_selector (Optional[str]): Capture a screenshot of a specific element using CSS Selector.
        original_status (Optional[bool]): Return the original HTTP status code from the target page. Default False.
        allowed_status_codes (Optional[str]): Returns content even if target page fails with specified status codes.
        json_response (Optional[bool]): Capture network requests in JSON format. Default False.
        screenshot_format (Optional[Literal["png", "jpeg"]]): Choose between png and jpeg formats for screenshots.
        screenshot_quality (Optional[int]): For JPEG format, set quality from 1 to 100.
        outputs (Optional[str]): Specify which data types to extract from the scraped HTML.

    """

    is_remote: bool = True
    api_key: str = Field(description="ZenRows API key")
    js_render: Optional[bool] = Field(
        default=False,
        description="Enable JavaScript rendering with a headless browser. Essential for modern web apps, SPAs, and sites with dynamic content.",
    )
    js_instructions: Optional[str] = Field(
        default=None,
        description="Execute custom JavaScript on the page to interact with elements, scroll, click buttons, or manipulate content.",
    )
    premium_proxy: Optional[bool] = Field(
        default=False,
        description="Use residential IPs to bypass anti-bot protection. Essential for accessing protected sites.",
    )
    proxy_country: Optional[str] = Field(
        default=None,
        description="Set the country of the IP used for the request (requires Premium Proxies). Use for accessing geo-restricted content.",
    )
    session_id: Optional[int] = Field(
        default=None,
        description="Maintain the same IP for multiple requests for up to 10 minutes. Essential for multi-step processes.",
    )
    custom_headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="Include custom headers in your request to mimic browser behavior.",
    )
    wait_for: Optional[str] = Field(
        default=None,
        description="Wait for a specific CSS Selector to appear in the DOM before returning content.",
    )
    wait: Optional[int] = Field(
        default=None, description="Wait a fixed amount of milliseconds after page load."
    )
    block_resources: Optional[str] = Field(
        default=None,
        description="Block specific resources (images, fonts, etc.) from loading to speed up scraping.",
    )
    response_type: Optional[Literal["markdown", "plaintext", "pdf"]] = Field(
        default=None,
        description="Convert HTML to other formats. Options: markdown, plaintext, pdf.",
    )
    css_extractor: Optional[str] = Field(
        default=None,
        description="Extract specific elements using CSS selectors (JSON format).",
    )
    autoparse: Optional[bool] = Field(
        default=False, description="Automatically extract structured data from HTML."
    )
    screenshot: Optional[str] = Field(
        default=None, description="Capture an above-the-fold screenshot of the page."
    )
    screenshot_fullpage: Optional[str] = Field(
        default=None, description="Capture a full-page screenshot."
    )
    screenshot_selector: Optional[str] = Field(
        default=None,
        description="Capture a screenshot of a specific element using CSS Selector.",
    )
    original_status: Optional[bool] = Field(
        default=False,
        description="Return the original HTTP status code from the target page.",
    )
    allowed_status_codes: Optional[str] = Field(
        default=None,
        description="Returns the content even if the target page fails with specified status codes.",
    )
    json_response: Optional[bool] = Field(
        default=False,
        description="Capture network requests in JSON format, including XHR or Fetch data.",
    )
    screenshot_format: Optional[Literal["png", "jpeg"]] = Field(
        default=None,
        description="Choose between png (default) and jpeg formats for screenshots.",
    )
    screenshot_quality: Optional[int] = Field(
        default=None,
        description="For JPEG format, set quality from 1 to 100.",
    )
    outputs: Optional[str] = Field(
        default=None,
        description="Specify which data types to extract from the scraped HTML.",
    )

    _base_url: str = PrivateAttr(default="https://api.zenrows.com/v1/")

    @field_validator("css_extractor")
    @classmethod
    def validate_css_extractor(cls, v):
        """Validate that css_extractor is valid JSON if provided."""
        if v is not None:
            try:
                json.loads(v)
            except json.JSONDecodeError:
                raise ValueError("css_extractor must be valid JSON")
        return v

    @field_validator("proxy_country")
    @classmethod
    def validate_proxy_country(cls, v):
        """Validate that proxy_country is a two-letter country code."""
        if v is not None and len(v) != 2:
            raise ValueError("proxy_country must be a two-letter country code")
        return v

    def __init__(self, **kwargs):
        """Initialize ZenRows Web Reader."""
        super().__init__(**kwargs)
        if not self.api_key:
            raise ValueError(
                "ZenRows API key is required. Get one at https://app.zenrows.com/register"
            )

    @classmethod
    def class_name(cls) -> str:
        """Get the name identifier of the class."""
        return "ZenRowsWebReader"

    def _prepare_request_params(
        self, url: str, extra_params: Optional[Dict] = None
    ) -> tuple[Dict[str, Any], Optional[Dict[str, str]]]:
        """Prepare request parameters for ZenRows API."""
        params = {"url": url, "apikey": self.api_key}

        # Add all configured parameters
        if self.js_render:
            params["js_render"] = self.js_render
        if self.js_instructions:
            params["js_instructions"] = self.js_instructions
        if self.premium_proxy:
            params["premium_proxy"] = self.premium_proxy
        if self.proxy_country:
            params["proxy_country"] = self.proxy_country
        if self.session_id:
            params["session_id"] = self.session_id
        if self.wait_for:
            params["wait_for"] = self.wait_for
        if self.wait:
            params["wait"] = self.wait
        if self.block_resources:
            params["block_resources"] = self.block_resources
        if self.response_type:
            params["response_type"] = self.response_type
        if self.css_extractor:
            params["css_extractor"] = self.css_extractor
        if self.autoparse:
            params["autoparse"] = self.autoparse
        if self.screenshot:
            params["screenshot"] = self.screenshot
        if self.screenshot_fullpage:
            params["screenshot_fullpage"] = self.screenshot_fullpage
        if self.screenshot_selector:
            params["screenshot_selector"] = self.screenshot_selector
        if self.original_status:
            params["original_status"] = self.original_status
        if self.allowed_status_codes:
            params["allowed_status_codes"] = self.allowed_status_codes
        if self.json_response:
            params["json_response"] = self.json_response
        if self.screenshot_format:
            params["screenshot_format"] = self.screenshot_format
        if self.screenshot_quality:
            params["screenshot_quality"] = self.screenshot_quality
        if self.outputs:
            params["outputs"] = self.outputs

        # Add any extra parameters for this specific request
        if extra_params:
            params.update(extra_params)

        # Auto-enable js_render for parameters that require JavaScript rendering
        js_required_params = [
            "screenshot",
            "screenshot_fullpage",
            "screenshot_selector",
            "js_instructions",
            "json_response",
            "wait",
            "wait_for",
        ]
        js_required = any(params.get(param) for param in js_required_params)

        if js_required:
            params["js_render"] = True

        # Special handling for screenshot variants
        screenshot_variants = ["screenshot_fullpage", "screenshot_selector"]
        if any(params.get(param) for param in screenshot_variants):
            params["screenshot"] = "true"

        # Auto-enable premium_proxy when proxy_country is specified
        if params.get("proxy_country"):
            params["premium_proxy"] = True

        # Handle custom headers
        request_headers = None
        if "custom_headers" in params and params["custom_headers"]:
            # Store the headers dictionary for the request
            request_headers = params["custom_headers"]
            # Set custom_headers to "true" to enable custom header support in the API
            params["custom_headers"] = "true"
        elif self.custom_headers:
            request_headers = self.custom_headers
            params["custom_headers"] = "true"
        else:
            # Remove custom_headers if not provided or empty
            params.pop("custom_headers", None)

        # Remove None values to avoid sending unnecessary parameters
        params = {k: v for k, v in params.items() if v is not None}

        return params, request_headers

    def _make_request(
        self, url: str, extra_params: Optional[Dict] = None
    ) -> requests.Response:
        """Make request to ZenRows API."""
        params, request_headers = self._prepare_request_params(url, extra_params)

        response = requests.get(
            self._base_url,
            params=params,
            headers=request_headers,
        )
        response.raise_for_status()
        return response

    def _extract_metadata(
        self, response: requests.Response, url: str
    ) -> Dict[str, Any]:
        """Extract metadata from ZenRows response."""
        metadata = {
            "source_url": url,
            "scraped_at": time.time(),
        }

        # Extract ZenRows specific headers
        if "X-Request-Cost" in response.headers:
            metadata["request_cost"] = float(response.headers["X-Request-Cost"])
        if "X-Request-Id" in response.headers:
            metadata["request_id"] = response.headers["X-Request-Id"]
        if "Zr-Final-Url" in response.headers:
            metadata["final_url"] = response.headers["Zr-Final-Url"]
        if "Concurrency-Remaining" in response.headers:
            metadata["concurrency_remaining"] = int(
                response.headers["Concurrency-Remaining"]
            )
        if "Concurrency-Limit" in response.headers:
            metadata["concurrency_limit"] = int(response.headers["Concurrency-Limit"])

        # Add response info
        metadata["status_code"] = response.status_code
        metadata["content_type"] = response.headers.get("Content-Type", "")
        metadata["content_length"] = len(response.content)

        # Add scraping configuration used
        metadata["zenrows_config"] = {
            "js_render": self.js_render,
            "premium_proxy": self.premium_proxy,
            "proxy_country": self.proxy_country,
            "session_id": self.session_id,
            "response_type": self.response_type,
        }

        return metadata

    def _process_response_content(self, response: requests.Response) -> str:
        """Process response content based on whether it's a screenshot or not."""
        # Handle screenshot responses
        screenshot_params = ["screenshot", "screenshot_fullpage", "screenshot_selector"]
        if any(getattr(self, param, None) for param in screenshot_params):
            return response.content

        # For all other responses, return text
        return response.text

    def load_data(
        self, urls: Union[str, List[str]], extra_params: Optional[Dict] = None, **kwargs
    ) -> List[Document]:
        """
        Load data from URLs using ZenRows API.

        Args:
            urls: Single URL string or list of URLs to scrape
            extra_params: Additional parameters for this specific request
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
                content = self._process_response_content(response)
                metadata = self._extract_metadata(response, url)

                # Create document
                document = Document(
                    text=content,
                    metadata=metadata,
                )
                documents.append(document)

            except Exception as e:
                # Create error document for failed URLs
                error_metadata = {
                    "source_url": url,
                    "error": str(e),
                    "scraped_at": time.time(),
                    "status": "failed",
                }
                error_document = Document(
                    text=f"Error scraping {url}: {e!s}",
                    metadata=error_metadata,
                )
                documents.append(error_document)

        return documents
