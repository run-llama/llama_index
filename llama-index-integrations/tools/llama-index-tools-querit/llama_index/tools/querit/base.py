"""Querit Search API tool for LlamaIndex."""

from typing import List, Optional

from llama_index.core.tools.tool_spec.base import BaseToolSpec

from querit import QueritClient
from querit.models.request import GeoFilter, SearchFilters, SearchRequest, SiteFilter


class QueritToolSpec(BaseToolSpec):
    """Querit Search API tool spec.

    Provides web search capabilities powered by the Querit Search API.
    Each method corresponds to one callable tool exposed to the LLM agent.
    """

    spec_functions = [
        "search",
        "search_with_language",
        "search_with_geo",
        "search_with_site_filter",
        "search_with_time_range",
    ]

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.querit.ai",
        timeout: float = 60.0,
    ) -> None:
        """Initialize the Querit tool spec.

        Args:
            api_key (str): Querit API key for authentication.
            base_url (str): Base API endpoint URL. Defaults to "https://api.querit.ai".
            timeout (float): HTTP request timeout in seconds. Defaults to 60.0.
        """
        super().__init__()
        self._client = QueritClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def _format_results(self, response: object) -> str:
        """Format a SearchResponse into a readable string."""
        if response.error_code and response.error_code != 200:  # type: ignore[union-attr]
            return f"Search error {response.error_code}: {response.error_msg}"  # type: ignore[union-attr]

        results = response.results  # type: ignore[union-attr]
        if not results:
            return "No results found."

        lines = [f"Search Results (search_id={response.search_id}):"]  # type: ignore[union-attr]
        for i, item in enumerate(results, start=1):
            lines.append(f"\n[{i}] {item.title or '(No title)'}")
            lines.append(f"    URL: {item.url or '(No URL)'}")
            if item.snippet:
                lines.append(f"    Snippet: {item.snippet}")
            if item.page_age:
                lines.append(f"    Age: {item.page_age}")
        return "\n".join(lines)

    def search(self, query: str, count: int = 10) -> str:
        """Perform a basic web search using the Querit Search API.

        Args:
            query (str): The search query text.
            count (int): Maximum number of results to return. Defaults to 10.

        Returns:
            str: Formatted search results including titles, URLs, and snippets.
        """
        req = SearchRequest(query=query, count=count)
        response = self._client.search(req)
        return self._format_results(response)

    def search_with_language(
        self, query: str, language: str, count: int = 10
    ) -> str:
        """Perform a web search filtered by language.

        Args:
            query (str): The search query text.
            language (str): Language to filter results by. Supported values:
                "english", "japanese", "korean", "german", "french",
                "spanish", "portuguese".
            count (int): Maximum number of results to return. Defaults to 10.

        Returns:
            str: Formatted search results filtered by the specified language.
        """
        req = SearchRequest(
            query=query,
            count=count,
            filters=SearchFilters(languages=[language]),
        )
        response = self._client.search(req)
        return self._format_results(response)

    def search_with_geo(
        self, query: str, country: str, count: int = 10
    ) -> str:
        """Perform a web search restricted to results from a specific country.

        Args:
            query (str): The search query text.
            country (str): Country to restrict results to. Supported values:
                "united states", "united kingdom", "japan", "germany", "france",
                "australia", "canada", "india", "brazil", "south korea",
                "spain", "mexico", "argentina", "colombia", "indonesia",
                "nigeria", "philippines".
            count (int): Maximum number of results to return. Defaults to 10.

        Returns:
            str: Formatted search results restricted to the specified country.
        """
        req = SearchRequest(
            query=query,
            count=count,
            filters=SearchFilters(geo=GeoFilter(countries=[country])),
        )
        response = self._client.search(req)
        return self._format_results(response)

    def search_with_site_filter(
        self,
        query: str,
        include_sites: Optional[List[str]] = None,
        exclude_sites: Optional[List[str]] = None,
        count: int = 10,
    ) -> str:
        """Perform a web search with site-level whitelist or blacklist filters.

        Args:
            query (str): The search query text.
            include_sites (Optional[List[str]]): List of domain names to restrict
                results to. Example: ["wikipedia.org", "bbc.com"].
            exclude_sites (Optional[List[str]]): List of domain names to exclude
                from results. Example: ["spam.com"].
            count (int): Maximum number of results to return. Defaults to 10.

        Returns:
            str: Formatted search results with site filters applied.
        """
        req = SearchRequest(
            query=query,
            count=count,
            filters=SearchFilters(
                sites=SiteFilter(include=include_sites, exclude=exclude_sites)
            ),
        )
        response = self._client.search(req)
        return self._format_results(response)

    def search_with_time_range(
        self, query: str, time_range: str, count: int = 10
    ) -> str:
        """Perform a web search filtered to a specific time range.

        Args:
            query (str): The search query text.
            time_range (str): Time range filter string. Examples:
                "d1" for past day, "w1" for past week, "m1" for past month,
                "m7" for past 7 months, "y1" for past year.
            count (int): Maximum number of results to return. Defaults to 10.

        Returns:
            str: Formatted search results filtered to the specified time range.
        """
        req = SearchRequest(
            query=query,
            count=count,
            filters=SearchFilters(time_range=time_range),
        )
        response = self._client.search(req)
        return self._format_results(response)
