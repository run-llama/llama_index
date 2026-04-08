"""MrScraper Web Reader."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Literal, Optional

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 120


def _get_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


class MrScraperWebReader(BasePydanticReader):
    """
    Read web pages using MrScraper API with AI-powered scraping capabilities.

    MrScraper provides:
    - Stealth browser rendering with bot detection evasion
    - AI-powered data extraction with natural-language instructions
    - Geolocation proxy support
    - Bulk scraping across multiple URLs
    - Manual and AI scraper management

    Args:
        api_token: MrScraper API token. Get yours at https://app.mrscraper.com.
        mode: The mode to run the loader in.
            Options:
            - ``"fetch_html"`` — Fetch rendered HTML via stealth browser (default).
            - ``"scrape"`` — Create an AI scraper and extract structured data.
            - ``"rerun_scraper"`` — Rerun an existing AI scraper on a new URL.
            - ``"bulk_rerun_ai_scraper"`` — Rerun AI scraper on multiple URLs.
            - ``"rerun_manual_scraper"`` — Rerun a manual scraper on a new URL.
            - ``"bulk_rerun_manual_scraper"`` — Rerun manual scraper on multiple URLs.
            - ``"get_all_results"`` — Retrieve paginated scraping results.
            - ``"get_result_by_id"`` — Retrieve a specific result by ID.

    Example::

        from llama_index.readers.web import MrScraperWebReader

        reader = MrScraperWebReader(api_token="YOUR_TOKEN", mode="fetch_html")
        documents = reader.load_data(url="https://example.com")

    """

    is_remote: bool = True
    api_token: str = Field(description="MrScraper API token")
    mode: str = Field(
        default="fetch_html",
        description="Operation mode: fetch_html, scrape, rerun_scraper, "
        "bulk_rerun_ai_scraper, rerun_manual_scraper, "
        "bulk_rerun_manual_scraper, get_all_results, get_result_by_id",
    )

    _client: Any = PrivateAttr(default=None)

    def __init__(self, api_token: str, mode: str = "fetch_html", **kwargs: Any) -> None:
        if not api_token:
            raise ValueError(
                "MrScraper API token is required. "
                "Get yours at https://app.mrscraper.com"
            )
        valid_modes = {
            "fetch_html",
            "scrape",
            "rerun_scraper",
            "bulk_rerun_ai_scraper",
            "rerun_manual_scraper",
            "bulk_rerun_manual_scraper",
            "get_all_results",
            "get_result_by_id",
        }
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of: {', '.join(sorted(valid_modes))}"
            )
        super().__init__(api_token=api_token, mode=mode, **kwargs)
        self._client = self._init_client(api_token)

    @staticmethod
    def _init_client(api_token: str) -> Any:
        try:
            from mrscraper import MrScraper
        except ImportError:
            raise ImportError(
                "mrscraper-sdk not found. Please run `pip install mrscraper-sdk`"
            )
        return MrScraper(token=api_token)

    @classmethod
    def class_name(cls) -> str:
        return "MrScraperWebReader"

    # ------------------------------------------------------------------
    # Async core methods — each wraps one SDK method
    # ------------------------------------------------------------------

    async def _fetch_html(
        self,
        url: str,
        *,
        timeout: int = _DEFAULT_TIMEOUT,
        geo_code: str = "US",
        block_resources: bool = False,
    ) -> Document:
        result = await self._client.fetch_html(
            url,
            timeout=timeout,
            geo_code=geo_code,
            block_resources=block_resources,
        )
        text = self._extract_text(result)
        metadata = {
            "source_url": url,
            "status_code": result.get("status_code"),
            "mode": "fetch_html",
            "geo_code": geo_code,
        }
        return Document(text=text, metadata=metadata)

    async def _create_scraper(
        self,
        url: str,
        message: str,
        *,
        agent: Literal["general", "listing", "map"] = "general",
        proxy_country: Optional[str] = None,
        max_depth: int = 2,
        max_pages: int = 50,
        limit: int = 1000,
        include_patterns: str = "",
        exclude_patterns: str = "",
    ) -> Document:
        result = await self._client.create_scraper(
            url,
            message,
            agent=agent,
            proxy_country=proxy_country,
            max_depth=max_depth,
            max_pages=max_pages,
            limit=limit,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
        text = self._extract_text(result)
        data = result.get("data", {})
        scraper_id = data.get("id") if isinstance(data, dict) else None
        metadata = {
            "source_url": url,
            "status_code": result.get("status_code"),
            "mode": "scrape",
            "agent": agent,
            "scraper_id": scraper_id,
            "message": message,
        }
        return Document(text=text, metadata=metadata)

    async def _rerun_scraper(
        self,
        scraper_id: str,
        url: str,
        *,
        max_depth: int = 2,
        max_pages: int = 50,
        limit: int = 1000,
        include_patterns: str = "",
        exclude_patterns: str = "",
    ) -> Document:
        result = await self._client.rerun_scraper(
            scraper_id,
            url,
            max_depth=max_depth,
            max_pages=max_pages,
            limit=limit,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
        text = self._extract_text(result)
        metadata = {
            "source_url": url,
            "status_code": result.get("status_code"),
            "mode": "rerun_scraper",
            "scraper_id": scraper_id,
        }
        return Document(text=text, metadata=metadata)

    async def _bulk_rerun_ai_scraper(
        self,
        scraper_id: str,
        urls: List[str],
    ) -> List[Document]:
        result = await self._client.bulk_rerun_ai_scraper(scraper_id, urls)
        text = self._extract_text(result)
        metadata = {
            "source_urls": urls,
            "status_code": result.get("status_code"),
            "mode": "bulk_rerun_ai_scraper",
            "scraper_id": scraper_id,
        }
        return [Document(text=text, metadata=metadata)]

    async def _rerun_manual_scraper(
        self,
        scraper_id: str,
        url: str,
    ) -> Document:
        result = await self._client.rerun_manual_scraper(scraper_id, url)
        text = self._extract_text(result)
        metadata = {
            "source_url": url,
            "status_code": result.get("status_code"),
            "mode": "rerun_manual_scraper",
            "scraper_id": scraper_id,
        }
        return Document(text=text, metadata=metadata)

    async def _bulk_rerun_manual_scraper(
        self,
        scraper_id: str,
        urls: List[str],
    ) -> List[Document]:
        result = await self._client.bulk_rerun_manual_scraper(scraper_id, urls)
        text = self._extract_text(result)
        metadata = {
            "source_urls": urls,
            "status_code": result.get("status_code"),
            "mode": "bulk_rerun_manual_scraper",
            "scraper_id": scraper_id,
        }
        return [Document(text=text, metadata=metadata)]

    async def _get_all_results(
        self,
        *,
        sort_field: Literal[
            "createdAt",
            "updatedAt",
            "id",
            "type",
            "url",
            "status",
            "error",
            "tokenUsage",
            "runtime",
        ] = "updatedAt",
        sort_order: Literal["ASC", "DESC"] = "DESC",
        page_size: int = 10,
        page: int = 1,
        search: Optional[str] = None,
        date_range_column: Optional[str] = None,
        start_at: Optional[str] = None,
        end_at: Optional[str] = None,
    ) -> List[Document]:
        result = await self._client.get_all_results(
            sort_field=sort_field,
            sort_order=sort_order,
            page_size=page_size,
            page=page,
            search=search,
            date_range_column=date_range_column,
            start_at=start_at,
            end_at=end_at,
        )
        data = result.get("data", {})
        items = data.get("data", []) if isinstance(data, dict) else []
        if isinstance(items, list) and items:
            documents: List[Document] = []
            for item in items:
                item_text = self._format_result_item(item)
                item_meta = {
                    "status_code": result.get("status_code"),
                    "mode": "get_all_results",
                    "result_id": item.get("id") if isinstance(item, dict) else None,
                }
                documents.append(Document(text=item_text, metadata=item_meta))
            return documents
        text = self._extract_text(result)
        return [
            Document(
                text=text,
                metadata={
                    "status_code": result.get("status_code"),
                    "mode": "get_all_results",
                },
            )
        ]

    async def _get_result_by_id(self, result_id: str) -> Document:
        result = await self._client.get_result_by_id(result_id)
        text = self._extract_text(result)
        metadata = {
            "status_code": result.get("status_code"),
            "mode": "get_result_by_id",
            "result_id": result_id,
        }
        return Document(text=text, metadata=metadata)

    # ------------------------------------------------------------------
    # Public async interface
    # ------------------------------------------------------------------

    async def aload_data(
        self,
        url: Optional[str] = None,
        urls: Optional[List[str]] = None,
        *,
        message: Optional[str] = None,
        scraper_id: Optional[str] = None,
        result_id: Optional[str] = None,
        timeout: int = _DEFAULT_TIMEOUT,
        geo_code: str = "US",
        block_resources: bool = False,
        agent: Literal["general", "listing", "map"] = "general",
        proxy_country: Optional[str] = None,
        max_depth: int = 2,
        max_pages: int = 50,
        limit: int = 1000,
        include_patterns: str = "",
        exclude_patterns: str = "",
        sort_field: Literal[
            "createdAt",
            "updatedAt",
            "id",
            "type",
            "url",
            "status",
            "error",
            "tokenUsage",
            "runtime",
        ] = "updatedAt",
        sort_order: Literal["ASC", "DESC"] = "DESC",
        page_size: int = 10,
        page: int = 1,
        search: Optional[str] = None,
        date_range_column: Optional[str] = None,
        start_at: Optional[str] = None,
        end_at: Optional[str] = None,
    ) -> List[Document]:
        """
        Asynchronously load data from MrScraper API.

        The parameters used depend on the ``mode`` set at initialization.

        Args:
            url: Target URL (used in fetch_html, scrape, rerun_scraper,
                 rerun_manual_scraper modes).
            urls: List of target URLs (used in bulk_rerun_ai_scraper,
                  bulk_rerun_manual_scraper modes).
            message: Natural-language extraction instructions (scrape mode).
            scraper_id: Scraper ID (rerun_scraper, bulk_rerun_ai_scraper,
                        rerun_manual_scraper, bulk_rerun_manual_scraper modes).
            result_id: Result ID (get_result_by_id mode).
            timeout: Page load timeout in seconds (fetch_html mode, default 120).
            geo_code: ISO country code for proxy geolocation (default ``"US"``).
            block_resources: Block images/CSS/fonts (fetch_html mode).
            agent: AI agent type: ``"general"``, ``"listing"``, or ``"map"``
                   (scrape mode).
            proxy_country: ISO country code for proxy (scrape mode).
            max_depth: Crawl depth for map agent (default 2).
            max_pages: Max pages for map agent (default 50).
            limit: Max records for map agent (default 1000).
            include_patterns: URL patterns to include (map agent).
            exclude_patterns: URL patterns to exclude (map agent).
            sort_field: Sort field for results listing (get_all_results mode).
            sort_order: Sort direction (get_all_results mode).
            page_size: Results per page (get_all_results mode).
            page: Page number (get_all_results mode).
            search: Text search query (get_all_results mode).
            date_range_column: Column for date range filter (get_all_results).
            start_at: ISO-8601 start date (get_all_results mode).
            end_at: ISO-8601 end date (get_all_results mode).

        Returns:
            List of Document objects containing scraped content and metadata.

        """
        if self.mode == "fetch_html":
            if url is None:
                raise ValueError("url is required for fetch_html mode.")
            doc = await self._fetch_html(
                url,
                timeout=timeout,
                geo_code=geo_code,
                block_resources=block_resources,
            )
            return [doc]

        elif self.mode == "scrape":
            if url is None:
                raise ValueError("url is required for scrape mode.")
            if message is None:
                raise ValueError("message is required for scrape mode.")
            doc = await self._create_scraper(
                url,
                message,
                agent=agent,
                proxy_country=proxy_country,
                max_depth=max_depth,
                max_pages=max_pages,
                limit=limit,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
            )
            return [doc]

        elif self.mode == "rerun_scraper":
            if scraper_id is None:
                raise ValueError("scraper_id is required for rerun_scraper mode.")
            if url is None:
                raise ValueError("url is required for rerun_scraper mode.")
            doc = await self._rerun_scraper(
                scraper_id,
                url,
                max_depth=max_depth,
                max_pages=max_pages,
                limit=limit,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
            )
            return [doc]

        elif self.mode == "bulk_rerun_ai_scraper":
            if scraper_id is None:
                raise ValueError(
                    "scraper_id is required for bulk_rerun_ai_scraper mode."
                )
            if urls is None or len(urls) == 0:
                raise ValueError(
                    "urls (non-empty list) is required for bulk_rerun_ai_scraper mode."
                )
            return await self._bulk_rerun_ai_scraper(scraper_id, urls)

        elif self.mode == "rerun_manual_scraper":
            if scraper_id is None:
                raise ValueError(
                    "scraper_id is required for rerun_manual_scraper mode."
                )
            if url is None:
                raise ValueError("url is required for rerun_manual_scraper mode.")
            doc = await self._rerun_manual_scraper(scraper_id, url)
            return [doc]

        elif self.mode == "bulk_rerun_manual_scraper":
            if scraper_id is None:
                raise ValueError(
                    "scraper_id is required for bulk_rerun_manual_scraper mode."
                )
            if urls is None or len(urls) == 0:
                raise ValueError(
                    "urls (non-empty list) is required for "
                    "bulk_rerun_manual_scraper mode."
                )
            return await self._bulk_rerun_manual_scraper(scraper_id, urls)

        elif self.mode == "get_all_results":
            return await self._get_all_results(
                sort_field=sort_field,
                sort_order=sort_order,
                page_size=page_size,
                page=page,
                search=search,
                date_range_column=date_range_column,
                start_at=start_at,
                end_at=end_at,
            )

        elif self.mode == "get_result_by_id":
            if result_id is None:
                raise ValueError("result_id is required for get_result_by_id mode.")
            doc = await self._get_result_by_id(result_id)
            return [doc]

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    # ------------------------------------------------------------------
    # Public sync interface
    # ------------------------------------------------------------------

    def load_data(
        self,
        url: Optional[str] = None,
        urls: Optional[List[str]] = None,
        *,
        message: Optional[str] = None,
        scraper_id: Optional[str] = None,
        result_id: Optional[str] = None,
        timeout: int = _DEFAULT_TIMEOUT,
        geo_code: str = "US",
        block_resources: bool = False,
        agent: Literal["general", "listing", "map"] = "general",
        proxy_country: Optional[str] = None,
        max_depth: int = 2,
        max_pages: int = 50,
        limit: int = 1000,
        include_patterns: str = "",
        exclude_patterns: str = "",
        sort_field: Literal[
            "createdAt",
            "updatedAt",
            "id",
            "type",
            "url",
            "status",
            "error",
            "tokenUsage",
            "runtime",
        ] = "updatedAt",
        sort_order: Literal["ASC", "DESC"] = "DESC",
        page_size: int = 10,
        page: int = 1,
        search: Optional[str] = None,
        date_range_column: Optional[str] = None,
        start_at: Optional[str] = None,
        end_at: Optional[str] = None,
    ) -> List[Document]:
        """
        Load data from the MrScraper API (synchronous wrapper).

        Accepts the same parameters as :meth:`aload_data`.  Internally runs the
        async implementation on an event loop.

        Returns:
            List of Document objects containing scraped content and metadata.

        """
        loop = _get_event_loop()
        return loop.run_until_complete(
            self.aload_data(
                url=url,
                urls=urls,
                message=message,
                scraper_id=scraper_id,
                result_id=result_id,
                timeout=timeout,
                geo_code=geo_code,
                block_resources=block_resources,
                agent=agent,
                proxy_country=proxy_country,
                max_depth=max_depth,
                max_pages=max_pages,
                limit=limit,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                sort_field=sort_field,
                sort_order=sort_order,
                page_size=page_size,
                page=page,
                search=search,
                date_range_column=date_range_column,
                start_at=start_at,
                end_at=end_at,
            )
        )

    # ------------------------------------------------------------------
    # SDK-aligned async methods (names match mrscraper-sdk / MrScraper client)
    # ------------------------------------------------------------------

    async def fetch_html(
        self,
        url: str,
        *,
        timeout: int = _DEFAULT_TIMEOUT,
        geo_code: str = "US",
        block_resources: bool = False,
    ) -> List[Document]:
        """Fetch rendered HTML of a page via the MrScraper stealth browser."""
        doc = await self._fetch_html(
            url, timeout=timeout, geo_code=geo_code, block_resources=block_resources
        )
        return [doc]

    async def create_scraper(
        self,
        url: str,
        message: str,
        *,
        agent: Literal["general", "listing", "map"] = "general",
        proxy_country: Optional[str] = None,
        max_depth: int = 2,
        max_pages: int = 50,
        limit: int = 1000,
        include_patterns: str = "",
        exclude_patterns: str = "",
    ) -> List[Document]:
        """Create an AI-powered scraper and run it immediately."""
        doc = await self._create_scraper(
            url,
            message,
            agent=agent,
            proxy_country=proxy_country,
            max_depth=max_depth,
            max_pages=max_pages,
            limit=limit,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
        return [doc]

    async def rerun_scraper(
        self,
        scraper_id: str,
        url: str,
        *,
        max_depth: int = 2,
        max_pages: int = 50,
        limit: int = 1000,
        include_patterns: str = "",
        exclude_patterns: str = "",
    ) -> List[Document]:
        """Rerun an existing AI scraper on a (new) URL."""
        doc = await self._rerun_scraper(
            scraper_id,
            url,
            max_depth=max_depth,
            max_pages=max_pages,
            limit=limit,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
        return [doc]

    async def bulk_rerun_ai_scraper(
        self,
        scraper_id: str,
        urls: List[str],
    ) -> List[Document]:
        """Rerun an AI scraper on multiple URLs in a single batch."""
        return await self._bulk_rerun_ai_scraper(scraper_id, urls)

    async def rerun_manual_scraper(
        self,
        scraper_id: str,
        url: str,
    ) -> List[Document]:
        """Rerun a manually configured scraper on a new URL."""
        doc = await self._rerun_manual_scraper(scraper_id, url)
        return [doc]

    async def bulk_rerun_manual_scraper(
        self,
        scraper_id: str,
        urls: List[str],
    ) -> List[Document]:
        """Rerun a manual scraper on multiple URLs in a single batch."""
        return await self._bulk_rerun_manual_scraper(scraper_id, urls)

    async def get_all_results(
        self,
        *,
        sort_field: Literal[
            "createdAt",
            "updatedAt",
            "id",
            "type",
            "url",
            "status",
            "error",
            "tokenUsage",
            "runtime",
        ] = "updatedAt",
        sort_order: Literal["ASC", "DESC"] = "DESC",
        page_size: int = 10,
        page: int = 1,
        search: Optional[str] = None,
        date_range_column: Optional[str] = None,
        start_at: Optional[str] = None,
        end_at: Optional[str] = None,
    ) -> List[Document]:
        """Retrieve a paginated list of all scraping results."""
        return await self._get_all_results(
            sort_field=sort_field,
            sort_order=sort_order,
            page_size=page_size,
            page=page,
            search=search,
            date_range_column=date_range_column,
            start_at=start_at,
            end_at=end_at,
        )

    async def get_result_by_id(self, result_id: str) -> List[Document]:
        """Retrieve full details of a specific scraping result by ID."""
        doc = await self._get_result_by_id(result_id)
        return [doc]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(result: Dict[str, Any]) -> str:
        data = result.get("data", "")
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            return json.dumps(data, indent=2, default=str)
        if isinstance(data, list):
            return json.dumps(data, indent=2, default=str)
        return str(data)

    @staticmethod
    def _format_result_item(item: Any) -> str:
        if isinstance(item, dict):
            return json.dumps(item, indent=2, default=str)
        return str(item)
