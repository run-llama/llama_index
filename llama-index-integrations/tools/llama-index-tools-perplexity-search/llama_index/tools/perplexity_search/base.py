"""Perplexity Search tool spec."""

import os
from typing import List, Literal, Optional

import requests

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

DEFAULT_ENDPOINT = "https://api.perplexity.ai/search"


class PerplexitySearchToolSpec(BaseToolSpec):
    """Perplexity Search tool spec.

    Search the web for up-to-date information using the Perplexity Search API.
    Returns ranked results with snippets, titles, URLs, and dates.
    """

    spec_functions = ["perplexity_search"]

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout: int = 30,
    ) -> None:
        """Initialize the tool spec.

        Args:
            api_key: Perplexity API key. If omitted, falls back to the
                ``PERPLEXITY_API_KEY`` environment variable, then to
                ``PPLX_API_KEY``.
            endpoint: Override the Search API endpoint (defaults to
                ``https://api.perplexity.ai/search``).
            timeout: Request timeout in seconds.
        """
        resolved_key = (
            api_key
            or os.environ.get("PERPLEXITY_API_KEY")
            or os.environ.get("PPLX_API_KEY")
        )
        if not resolved_key:
            raise ValueError(
                "A Perplexity API key is required. Pass api_key=... or set "
                "PERPLEXITY_API_KEY (or PPLX_API_KEY) in the environment."
            )
        self._api_key = resolved_key
        self._endpoint = endpoint
        self._timeout = timeout

    def perplexity_search(
        self,
        query: str,
        max_results: int = 5,
        search_domain_filter: Optional[List[str]] = None,
        search_recency_filter: Optional[
            Literal["hour", "day", "week", "month", "year"]
        ] = None,
    ) -> List[Document]:
        """Search the web with the Perplexity Search API.

        Returns ranked results with snippets, titles, URLs, and dates.

        Args:
            query: The natural-language search query.
            max_results: Maximum number of results to return (default 5).
            search_domain_filter: Optional list of domains. Use bare domains to
                allowlist (e.g. ``"nytimes.com"``) or prefix with ``-`` to
                denylist (e.g. ``"-pinterest.com"``). Do not mix the two — the
                API treats the list as either allow or deny, never both.
            search_recency_filter: Restrict results to a recency window. One of
                ``"hour"``, ``"day"``, ``"week"``, ``"month"``, ``"year"``.

        Returns:
            A list of ``Document`` objects, one per search result. Each
            document's ``text`` is the result snippet and its ``metadata``
            contains ``title``, ``url``, and ``date``.
        """
        payload = {"query": query, "max_results": max_results}
        if search_domain_filter is not None:
            payload["search_domain_filter"] = search_domain_filter
        if search_recency_filter is not None:
            payload["search_recency_filter"] = search_recency_filter

        response = requests.post(
            self._endpoint,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self._timeout,
        )
        response.raise_for_status()
        body = response.json()

        return [
            Document(
                text=result.get("snippet", ""),
                extra_info={
                    "title": result.get("title"),
                    "url": result.get("url"),
                    "date": result.get("date"),
                },
            )
            for result in body.get("results", [])
        ]
