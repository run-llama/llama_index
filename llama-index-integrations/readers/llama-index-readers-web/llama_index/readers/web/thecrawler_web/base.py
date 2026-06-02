"""TheCrawler Web Reader."""

import os
from typing import Any, Callable, Dict, List, Optional

import requests

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document


_DEFAULT_API_URL = "https://www.miaibot.ai/api/v1"


class TheCrawlerWebReader(BasePydanticReader):
    """
    Load web pages as ``Document`` instances using the TheCrawler hosted API.

    TheCrawler returns boilerplate-stripped markdown plus rich page metadata
    (title, description, status code, structured error type, etc.) per URL.
    PDF and DOCX URLs are auto-handled by the server. The reader does not raise
    on a per-page failure: failed pages are returned as ``Document`` instances
    with empty text and ``status="error"`` plus a structured ``error_type`` in
    metadata, so caller code can branch on it.

    Args:
        api_key: TheCrawler API key (``mai_live_...``). If omitted, the value
            of the ``THECRAWLER_API_KEY`` environment variable is used.
        api_url: Base URL of the TheCrawler API. Defaults to
            ``https://www.miaibot.ai/api/v1``. Override to point at a
            self-hosted ``thecrawler-api`` instance.
        params: Extra options forwarded to the ``/crawl`` request body. See
            the TheCrawler README for the full option list (e.g.
            ``{"usePlaywright": True, "requestTimeoutSecs": 60}``).
        timeout: Per-request timeout in seconds for the HTTP call to
            TheCrawler. Default 120.

    Example:
        >>> from llama_index.readers.web import TheCrawlerWebReader
        >>> reader = TheCrawlerWebReader(api_key="mai_live_...")
        >>> docs = reader.load_data(urls=["https://example.com"])

    """

    api_key: str
    api_url: str = _DEFAULT_API_URL
    params: Optional[Dict[str, Any]] = None
    timeout: int = 120

    _metadata_fn: Optional[Callable[[str], Dict]] = PrivateAttr()

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: int = 120,
    ) -> None:
        """Initialize with parameters."""
        resolved_key = api_key or os.getenv("THECRAWLER_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "TheCrawler API key is required. Pass api_key=... or set the "
                "THECRAWLER_API_KEY environment variable."
            )
        super().__init__(
            api_key=resolved_key,
            api_url=(api_url or _DEFAULT_API_URL).rstrip("/"),
            params=params or {},
            timeout=timeout,
        )

    @classmethod
    def class_name(cls) -> str:
        return "TheCrawlerWebReader"

    def _crawl_endpoint(self) -> str:
        return f"{self.api_url}/crawl"

    def _post_crawl(self, body: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            self._crawl_endpoint(),
            headers=headers,
            json=body,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _page_to_document(page: Dict[str, Any]) -> Document:
        url = page.get("url", "")
        text = page.get("markdown") or page.get("text") or ""

        metadata: Dict[str, Any] = {
            "source": "thecrawler",
            "url": url,
            "title": page.get("title"),
            "description": page.get("description"),
            "language": page.get("language"),
            "canonical_url": page.get("canonicalUrl"),
            "status": page.get("status"),
            "status_code": page.get("statusCode"),
            "content_type": page.get("contentType"),
            "response_time_ms": page.get("responseTimeMs"),
            "scraped_at": page.get("scrapedAt"),
            "from_cache": page.get("fromCache"),
        }

        if page.get("status") == "error":
            metadata["error"] = page.get("error")
            metadata["error_type"] = page.get("errorType")
            metadata["error_retryable"] = page.get("errorRetryable")

        # Drop keys with ``None`` values to keep metadata tidy.
        metadata = {k: v for k, v in metadata.items() if v is not None}
        return Document(text=text, metadata=metadata)

    def load_data(self, urls: List[str]) -> List[Document]:
        """
        Crawl one or more URLs and return ``Document`` instances in input order.

        Args:
            urls: List of absolute URLs to crawl.

        Returns:
            A list of ``Document`` objects, one per input URL. URLs that
            failed to crawl are returned with empty text and the error
            details in metadata (``error_type``, ``error``, ``error_retryable``).

        """
        if not urls:
            raise ValueError("`urls` must be a non-empty list of URLs.")

        body: Dict[str, Any] = {
            "urls": list(urls),
            "extractMarkdown": True,
            "stripBoilerplate": True,
        }
        # User-supplied params override defaults.
        if self.params:
            body.update(self.params)

        result = self._post_crawl(body)
        pages = result.get("pages", []) if isinstance(result, dict) else []
        return [self._page_to_document(page) for page in pages]
