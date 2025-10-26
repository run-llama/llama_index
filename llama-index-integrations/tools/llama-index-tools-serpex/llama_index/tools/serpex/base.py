"""SERPEX search tool specification."""

import os
from typing import Any, Dict, List, Optional

import requests

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class SerpexToolSpec(BaseToolSpec):
    """
    SERPEX tool spec for web search.

    This tool allows you to search the web using the SERPEX API and get
    real-time search results from multiple search engines including Google,
    Bing, DuckDuckGo, Brave, Yahoo, and Yandex.

    SERPEX provides fast, reliable search results via API, perfect for
    AI applications, RAG systems, and data analytics.

    Args:
        api_key (Optional[str]): SERPEX API key. If not provided, will look
            for SERPEX_API_KEY environment variable.
        engine (str): Default search engine to use. Options: 'auto' (default),
            'google', 'bing', 'duckduckgo', 'brave', 'yahoo', 'yandex'.

    Examples:
        >>> from llama_index.tools.serpex import SerpexToolSpec
        >>> tool = SerpexToolSpec(api_key="your_api_key")
        >>> results = tool.search("latest AI news")
        >>> for doc in results:
        ...     print(doc.text)
    """

    spec_functions = ["search"]

    def __init__(
        self,
        api_key: Optional[str] = None,
        engine: str = "auto",
    ) -> None:
        """
        Initialize SERPEX tool.

        Args:
            api_key: SERPEX API key. If not provided, reads from
                SERPEX_API_KEY environment variable.
            engine: Default search engine ('auto', 'google', 'bing', etc.).

        Raises:
            ValueError: If API key is not provided and not found in environment.
        """
        self.api_key = api_key or os.environ.get("SERPEX_API_KEY")

        if not self.api_key:
            raise ValueError(
                "SERPEX_API_KEY not found. Please set it as an environment "
                "variable or pass it as an argument. "
                "Get your API key at: https://serpex.dev/dashboard"
            )

        self.base_url = "https://api.serpex.dev/api/search"
        self.engine = engine

    def search(
        self,
        query: str,
        num_results: int = 10,
        engine: Optional[str] = None,
        time_range: Optional[str] = None,
    ) -> List[Document]:
        """
        Search the web using SERPEX API.

        This function queries the specified search engine and returns structured
        results containing titles, URLs, and snippets.

        Args:
            query: Search query string.
            num_results: Number of results to return (default: 10, max: 100).
            engine: Override default search engine. Options: 'auto', 'google',
                'bing', 'duckduckgo', 'brave', 'yahoo', 'yandex'.
            time_range: Filter results by time. Options: 'day', 'week',
                'month', 'year'.

        Returns:
            List of Document objects containing search results.
            Each document's text contains formatted search results.

        Examples:
            >>> tool = SerpexToolSpec(api_key="your_key")
            >>> results = tool.search("LlamaIndex tutorial", num_results=5)
            >>> for doc in results:
            ...     print(doc.text)

            >>> # Search with specific engine
            >>> results = tool.search(
            ...     "privacy focused browser",
            ...     engine="duckduckgo",
            ...     num_results=5
            ... )

            >>> # Search with time filter
            >>> results = tool.search(
            ...     "AI news",
            ...     time_range="day",
            ...     num_results=10
            ... )
        """
        params: Dict[str, Any] = {
            "q": query,
            "engine": engine or self.engine,
            "category": "web",
        }

        if num_results:
            params["num"] = min(num_results, 100)  # Cap at 100

        if time_range:
            params["time_range"] = time_range

        try:
            response = requests.get(
                self.base_url,
                params=params,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                },
                timeout=30,
            )
            response.raise_for_status()

            data = response.json()

            # Extract results from the response
            results_list = data.get("results", [])

            if not results_list:
                return [Document(text="No search results found.")]

            # Format results as readable text
            formatted_results = []
            for i, result in enumerate(results_list[:num_results], 1):
                title = result.get("title", "No title")
                url = result.get("url", "")
                snippet = result.get("snippet", "No description available")

                formatted_results.append(
                    f"{i}. {title}\n" f"   URL: {url}\n" f"   {snippet}\n"
                )

            results_text = f"Search results for '{query}':\n\n"
            results_text += "\n".join(formatted_results)

            # Add metadata
            metadata = data.get("metadata", {})
            if metadata:
                num_results = metadata.get("number_of_results", 0)
                response_time = metadata.get("response_time", 0)
                results_text += (
                    f"\n\n(Found {num_results} results in {response_time}ms)"
                )

            return [Document(text=results_text)]

        except requests.exceptions.RequestException as e:
            error_msg = f"Error calling SERPEX API: {str(e)}"
            return [Document(text=error_msg)]
