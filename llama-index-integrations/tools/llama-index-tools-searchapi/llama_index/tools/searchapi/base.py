"""SearchApi.io tool spec.

SearchApi.io is a real-time SERP API that provides structured search results
from Google Search, Google News, Google Scholar, Google Images, and 50+
other search engines.

API Docs: https://www.searchapi.io/docs/google
"""

import os
from typing import Any, Dict, List, Optional

import httpx

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

DEFAULT_BASE_URL = "https://www.searchapi.io/api/v1/search"


class SearchApiToolSpec(BaseToolSpec):
    """SearchApi.io tool spec.

    Provides access to SearchApi.io for performing real-time web searches
    across multiple Google engines (web, news, scholar, images).

    Each method returns a list of ``Document`` objects with structured
    metadata in ``extra_info``, ready for downstream LlamaIndex pipelines.

    Args:
        api_key: SearchApi.io API key.  Falls back to the
            ``SEARCHAPI_API_KEY`` environment variable when omitted.
        base_url: Override the API endpoint (useful for testing/proxies).
    """

    spec_functions = [
        "search",
        "news_search",
        "scholar_search",
        "image_search",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """Initialize with parameters."""
        self.api_key = api_key or os.environ.get("SEARCHAPI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "SearchApi.io API key is required.  Pass it as `api_key` or "
                "set the SEARCHAPI_API_KEY environment variable."
            )
        self.base_url = base_url or DEFAULT_BASE_URL

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an HTTP GET request against the SearchApi.io endpoint."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        response = httpx.get(
            self.base_url,
            headers=headers,
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _build_params(
        query: str,
        engine: str,
        num: int,
        gl: Optional[str],
        hl: Optional[str],
        location: Optional[str],
    ) -> Dict[str, Any]:
        """Build the query-string parameters dict."""
        params: Dict[str, Any] = {
            "q": query,
            "engine": engine,
            "num": num,
        }
        if gl:
            params["gl"] = gl
        if hl:
            params["hl"] = hl
        if location:
            params["location"] = location
        return params

    # ------------------------------------------------------------------
    # Public tool methods
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        num: int = 10,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
        location: Optional[str] = None,
    ) -> List[Document]:
        """Search the web using Google via SearchApi.io.

        Args:
            query: The search query string.
            num: Maximum number of results to return (default 10).
            gl: Two-letter country code for geo-targeting (e.g. ``"us"``).
            hl: Two-letter language code (e.g. ``"en"``).
            location: Free-text location string (e.g. ``"New York, NY"``).

        Returns:
            A list of ``Document`` objects.  Each document's ``text`` is the
            result snippet and ``extra_info`` contains ``title``, ``link``,
            and ``position``.
        """
        params = self._build_params(query, "google", num, gl, hl, location)
        data = self._request(params)

        documents: List[Document] = []
        for result in data.get("organic_results", []):
            doc = Document(
                text=result.get("snippet", ""),
                metadata={
                    "title": result.get("title"),
                    "link": result.get("link"),
                    "position": result.get("position"),
                },
            )
            documents.append(doc)
        return documents

    def news_search(
        self,
        query: str,
        num: int = 10,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
        location: Optional[str] = None,
    ) -> List[Document]:
        """Search Google News via SearchApi.io.

        Args:
            query: The news search query string.
            num: Maximum number of results to return (default 10).
            gl: Two-letter country code for geo-targeting.
            hl: Two-letter language code.
            location: Free-text location string.

        Returns:
            A list of ``Document`` objects.  ``extra_info`` includes
            ``title``, ``link``, ``source``, and ``date``.
        """
        params = self._build_params(query, "google_news", num, gl, hl, location)
        data = self._request(params)

        documents: List[Document] = []
        for result in data.get("organic_results", []):
            doc = Document(
                text=result.get("snippet", ""),
                metadata={
                    "title": result.get("title"),
                    "link": result.get("link"),
                    "source": result.get("source"),
                    "date": result.get("date"),
                },
            )
            documents.append(doc)
        return documents

    def scholar_search(
        self,
        query: str,
        num: int = 10,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
        location: Optional[str] = None,
    ) -> List[Document]:
        """Search Google Scholar via SearchApi.io.

        Args:
            query: The academic search query string.
            num: Maximum number of results to return (default 10).
            gl: Two-letter country code for geo-targeting.
            hl: Two-letter language code.
            location: Free-text location string.

        Returns:
            A list of ``Document`` objects.  ``extra_info`` includes
            ``title``, ``link``, ``publication_info``, and ``cited_by_count``.
        """
        params = self._build_params(query, "google_scholar", num, gl, hl, location)
        data = self._request(params)

        documents: List[Document] = []
        for result in data.get("organic_results", []):
            pub_info = result.get("publication_info", {})
            inline_links = result.get("inline_links", {})
            cited_by = inline_links.get("cited_by", {})

            doc = Document(
                text=result.get("snippet", ""),
                metadata={
                    "title": result.get("title"),
                    "link": result.get("link"),
                    "publication_info": pub_info.get("summary", ""),
                    "cited_by_count": cited_by.get("total", 0),
                },
            )
            documents.append(doc)
        return documents

    def image_search(
        self,
        query: str,
        num: int = 10,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
        location: Optional[str] = None,
    ) -> List[Document]:
        """Search Google Images via SearchApi.io.

        Args:
            query: The image search query string.
            num: Maximum number of results to return (default 10).
            gl: Two-letter country code for geo-targeting.
            hl: Two-letter language code.
            location: Free-text location string.

        Returns:
            A list of ``Document`` objects.  ``extra_info`` includes
            ``link``, ``original`` (full-size image URL), and ``source``.
        """
        params = self._build_params(query, "google_images", num, gl, hl, location)
        data = self._request(params)

        documents: List[Document] = []
        for result in data.get("images_results", []):
            doc = Document(
                text=result.get("title", ""),
                metadata={
                    "link": result.get("link"),
                    "original": result.get("original"),
                    "source": result.get("source"),
                },
            )
            documents.append(doc)
        return documents
