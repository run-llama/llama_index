"""Perplexity Search API client."""

from __future__ import annotations

import os
from importlib import metadata
from typing import Any, Dict, List, Literal, Optional

import httpx
import requests

DEFAULT_ENDPOINT = "https://api.perplexity.ai/search"
DEFAULT_TIMEOUT = 30
RecencyFilter = Literal["hour", "day", "week", "month", "year"]


def _get_package_version() -> str:
    try:
        return metadata.version("llama-index-llms-perplexity")
    except metadata.PackageNotFoundError:
        return "unknown"


class PerplexitySearch:
    """
    Low-level client for the Perplexity Search API.

    The Search API is a standalone Perplexity product that returns ranked
    web search results. It is distinct from the Chat/Sonar completions API.

    Authentication falls back to the ``PERPLEXITY_API_KEY`` environment
    variable, then ``PPLX_API_KEY``.

    Example:
        >>> client = PerplexitySearch(api_key="...")
        >>> results = client.search("latest AI research", max_results=5)
        >>> for r in results:
        ...     print(r["title"], r["url"])

    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        resolved = (
            api_key or os.getenv("PERPLEXITY_API_KEY") or os.getenv("PPLX_API_KEY")
        )
        if not resolved:
            raise ValueError(
                "Perplexity API key not provided. Pass api_key=... or set the "
                "PERPLEXITY_API_KEY (or PPLX_API_KEY) environment variable."
            )
        self.api_key = resolved
        self.endpoint = endpoint
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Pplx-Integration": f"llamaindex/{_get_package_version()}",
        }

    def _build_payload(
        self,
        query: str,
        max_results: int,
        search_domain_filter: Optional[List[str]],
        search_recency_filter: Optional[RecencyFilter],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"query": query, "max_results": max_results}
        if search_domain_filter is not None:
            payload["search_domain_filter"] = search_domain_filter
        if search_recency_filter is not None:
            payload["search_recency_filter"] = search_recency_filter
        return payload

    @staticmethod
    def _extract_results(data: Any) -> List[Dict[str, Any]]:
        if isinstance(data, dict) and isinstance(data.get("results"), list):
            return data["results"]
        if isinstance(data, list):
            return data
        return []

    def search(
        self,
        query: str,
        max_results: int = 5,
        search_domain_filter: Optional[List[str]] = None,
        search_recency_filter: Optional[RecencyFilter] = None,
    ) -> List[Dict[str, Any]]:
        """Run a synchronous Perplexity search and return the raw result dicts."""
        payload = self._build_payload(
            query, max_results, search_domain_filter, search_recency_filter
        )
        response = requests.post(
            self.endpoint,
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return self._extract_results(response.json())

    async def asearch(
        self,
        query: str,
        max_results: int = 5,
        search_domain_filter: Optional[List[str]] = None,
        search_recency_filter: Optional[RecencyFilter] = None,
    ) -> List[Dict[str, Any]]:
        """Async variant of :meth:`search` using ``httpx.AsyncClient``."""
        payload = self._build_payload(
            query, max_results, search_domain_filter, search_recency_filter
        )
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.endpoint, headers=self._headers(), json=payload
            )
            response.raise_for_status()
            return self._extract_results(response.json())
