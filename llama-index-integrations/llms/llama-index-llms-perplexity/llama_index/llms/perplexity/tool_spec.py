"""LlamaIndex tool spec for the Perplexity Search API."""

from __future__ import annotations

from typing import List, Literal, Optional

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

from llama_index.llms.perplexity.search import PerplexitySearch

RecencyFilter = Literal["hour", "day", "week", "month", "year"]


class PerplexitySearchToolSpec(BaseToolSpec):
    """
    Tool spec wrapping the Perplexity Search API for use with LlamaIndex agents.

    Exposes a single tool, ``perplexity_search``, that calls the Search API
    and returns a list of :class:`~llama_index.core.schema.Document` results
    (text=snippet, metadata={title, url, date}).
    """

    spec_functions = ["perplexity_search"]

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: str = "https://api.perplexity.ai/search",
        timeout: int = 30,
    ) -> None:
        self.client = PerplexitySearch(
            api_key=api_key, endpoint=endpoint, timeout=timeout
        )

    def perplexity_search(
        self,
        query: str,
        max_results: int = 5,
        search_domain_filter: Optional[List[str]] = None,
        search_recency_filter: Optional[RecencyFilter] = None,
    ) -> List[Document]:
        """
        Search the web using the Perplexity Search API.

        Args:
            query: The search query.
            max_results: Maximum number of results to return.
            search_domain_filter: Optional list of domains to allow or deny.
                Use ``-domain.com`` to deny. Do not mix allowlist and denylist.
            search_recency_filter: Optional recency filter — one of
                ``hour``, ``day``, ``week``, ``month``, ``year``.

        Returns:
            A list of :class:`Document` objects.

        """
        results = self.client.search(
            query=query,
            max_results=max_results,
            search_domain_filter=search_domain_filter,
            search_recency_filter=search_recency_filter,
        )
        documents: List[Document] = []
        for item in results:
            text = item.get("snippet", "") or ""
            metadata = {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "date": item.get("date", ""),
            }
            documents.append(Document(text=text, metadata=metadata))
        return documents
