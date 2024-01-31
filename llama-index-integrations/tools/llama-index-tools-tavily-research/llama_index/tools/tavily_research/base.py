"""Tavily tool spec."""

from typing import Optional, List
from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class TavilyToolSpec(BaseToolSpec):
    """Tavily tool spec."""

    spec_functions = [
        "search",
    ]

    def __init__(self, api_key: str) -> None:
        """Initialize with parameters."""
        from tavily import TavilyClient

        self.client = TavilyClient(api_key=api_key)

    def search(self, query: str, max_results: Optional[int] = 6) -> List[Document]:
        """
        Run query through Tavily Search and return metadata.

        Args:
            query: The query to search for.
            max_results: The maximum number of results to return.

        Returns:
            results: A list of dictionaries containing the results:
                url: The url of the result.
                content: The content of the result.

        """
        response = self.client.search(
            query, max_results=max_results, search_depth="advanced"
        )
        return [
            Document(text=result["content"], extra_info={"url": result["url"]})
            for result in response["results"]
        ]
