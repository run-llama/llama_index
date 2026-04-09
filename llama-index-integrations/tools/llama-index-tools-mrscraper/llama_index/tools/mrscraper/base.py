"""MrScraper tool spec for LlamaIndex."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class MrScraperToolSpec(BaseToolSpec):
    """
    MrScraper tool spec for web scraping and AI-powered data extraction.

    Uses the MrScraper async SDK under the hood.  All public methods exposed as
    tools are coroutines so that LlamaIndex agents can call them with ``await``.
    """

    spec_functions = [
        "fetch_html",
        "create_scraper",
        "rerun_scraper",
        "bulk_rerun_ai_scraper",
        "rerun_manual_scraper",
        "bulk_rerun_manual_scraper",
        "get_all_results",
        "get_result_by_id",
    ]

    def __init__(
        self,
        api_key: str,
        *,
        verbose: bool = False,
    ) -> None:
        """
        Initialize MrScraperToolSpec.

        Args:
            api_key: Your MrScraper API token.
                     Get yours at https://app.mrscraper.com.
            verbose: When ``True`` extra diagnostic information is printed.

        """
        from mrscraper import MrScraper

        self._client = MrScraper(token=api_key)
        self._verbose = verbose

    async def fetch_html(
        self,
        url: str,
        timeout: int = 120,
        geo_code: str = "US",
        block_resources: bool = False,
    ) -> Document:
        """
        Fetch the rendered HTML of a page via the MrScraper stealth browser.

        The service handles JavaScript rendering, bot-detection evasion, and
        optional geolocation proxying.

        Args:
            url: Target URL to scrape.
            timeout: Maximum seconds to wait for the page to load (default 120).
            geo_code: ISO country code for proxy-based geolocation (default ``"US"``).
            block_resources: Block images/CSS/fonts to speed up the request.

        Returns:
            Document containing the raw HTML of the page.

        """
        if self._verbose:
            print(f"[MrScraper] Fetching HTML: {url}")

        return await self._client.fetch_html(
            url,
            timeout=timeout,
            geo_code=geo_code,
            block_resources=block_resources,
        )

    async def create_scraper(
        self,
        url: str,
        message: str,
        agent: Literal["general", "listing", "map"] = "general",
        proxy_country: Optional[str] = None,
        max_depth: int = 2,
        max_pages: int = 50,
        limit: int = 1000,
        include_patterns: str = "",
        exclude_patterns: str = "",
    ) -> Dict[str, Any]:
        """
        Create an AI-powered scraper and run it immediately.

        The scraper uses natural-language instructions to understand the page
        structure and extract the requested data automatically.

        Args:
            url: Target URL to scrape.
            message: Natural-language description of what to extract
                     (e.g. ``"Extract all product names and prices"``).
            agent: AI agent type — ``"general"`` (default), ``"listing"``, or
                   ``"map"`` (crawls sub-pages).
            proxy_country: ISO country code for proxy selection.
            max_depth: *(map only)* Crawl depth (default 2).
            max_pages: *(map only)* Max pages to process (default 50).
            limit: *(map only)* Max records to extract (default 1000).
            include_patterns: *(map only)* ``||``-separated URL include patterns.
            exclude_patterns: *(map only)* ``||``-separated URL exclude patterns.

        Returns:
            Dict with ``status_code``, ``data`` (scraper info incl. ID), and
            ``headers``.

        """
        if self._verbose:
            print(f"[MrScraper] Creating scraper for: {url}")

        return await self._client.create_scraper(
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

    async def rerun_scraper(
        self,
        scraper_id: str,
        url: str,
        max_depth: int = 2,
        max_pages: int = 50,
        limit: int = 1000,
        include_patterns: str = "",
        exclude_patterns: str = "",
    ) -> Dict[str, Any]:
        """
        Rerun an existing AI scraper on a (new) URL.

        Args:
            scraper_id: ID of the scraper to rerun.
            url: Target URL (can differ from the original).
            max_depth: *(map only)* Crawl depth (default 2).
            max_pages: *(map only)* Max pages (default 50).
            limit: *(map only)* Max records (default 1000).
            include_patterns: *(map only)* ``||``-separated include patterns.
            exclude_patterns: *(map only)* ``||``-separated exclude patterns.

        Returns:
            Dict with ``status_code``, ``data``, and ``headers``.

        """
        if self._verbose:
            print(f"[MrScraper] Rerunning scraper {scraper_id} on: {url}")

        return await self._client.rerun_scraper(
            scraper_id,
            url,
            max_depth=max_depth,
            max_pages=max_pages,
            limit=limit,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )

    async def bulk_rerun_ai_scraper(
        self,
        scraper_id: str,
        urls: List[str],
    ) -> Dict[str, Any]:
        """
        Rerun an AI scraper on multiple URLs in a single batch.

        More efficient than calling ``rerun_scraper`` in a loop because all
        URLs are dispatched in parallel server-side.

        Args:
            scraper_id: ID of the AI scraper to rerun.
            urls: List of target URLs (at least one).

        Returns:
            Dict with ``status_code``, ``data``, and ``headers``.

        """
        if self._verbose:
            print(f"[MrScraper] Bulk rerun AI scraper {scraper_id} on {len(urls)} URLs")

        return await self._client.bulk_rerun_ai_scraper(scraper_id, urls)

    async def rerun_manual_scraper(
        self,
        scraper_id: str,
        url: str,
    ) -> Dict[str, Any]:
        """
        Rerun a manually configured scraper on a new URL.

        Use this for scrapers built with custom CSS selectors or XPath rules
        via the MrScraper dashboard, *not* for AI scrapers.

        Args:
            scraper_id: ID of the manual scraper.
            url: Target URL.

        Returns:
            Dict with ``status_code``, ``data``, and ``headers``.

        """
        if self._verbose:
            print(f"[MrScraper] Rerunning manual scraper {scraper_id} on: {url}")

        return await self._client.rerun_manual_scraper(scraper_id, url)

    async def bulk_rerun_manual_scraper(
        self,
        scraper_id: str,
        urls: List[str],
    ) -> Dict[str, Any]:
        """
        Rerun a manual scraper on multiple URLs in a single batch.

        More efficient than calling ``rerun_manual_scraper`` multiple times.

        Args:
            scraper_id: ID of the manual scraper.
            urls: List of target URLs (at least one).

        Returns:
            Dict with ``status_code``, ``data`` (bulk job info), and ``headers``.

        """
        if self._verbose:
            print(
                f"[MrScraper] Bulk rerun manual scraper {scraper_id} "
                f"on {len(urls)} URLs"
            )

        return await self._client.bulk_rerun_manual_scraper(scraper_id, urls)

    async def get_all_results(
        self,
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
    ) -> Dict[str, Any]:
        """
        Retrieve a paginated list of all scraping results.

        Args:
            sort_field: Field to sort by (default ``"updatedAt"``).
            sort_order: ``"ASC"`` or ``"DESC"`` (default ``"DESC"``).
            page_size: Results per page (default 10).
            page: 1-indexed page number (default 1).
            search: Free-text search query (optional).
            date_range_column: Column to filter by date range (optional).
            start_at: ISO-8601 start date (optional).
            end_at: ISO-8601 end date (optional).

        Returns:
            Dict with ``status_code``, ``data`` (paginated results), and
            ``headers``.

        """
        if self._verbose:
            print(f"[MrScraper] Fetching results page {page}")

        return await self._client.get_all_results(
            sort_field=sort_field,
            sort_order=sort_order,
            page_size=page_size,
            page=page,
            search=search,
            date_range_column=date_range_column,
            start_at=start_at,
            end_at=end_at,
        )

    async def get_result_by_id(
        self,
        result_id: str,
    ) -> Dict[str, Any]:
        """
        Retrieve full details of a specific scraping result.

        Args:
            result_id: Unique identifier of the result.

        Returns:
            Dict with ``status_code``, ``data`` (complete result object), and
            ``headers``.

        """
        if self._verbose:
            print(f"[MrScraper] Fetching result: {result_id}")

        return await self._client.get_result_by_id(result_id)
