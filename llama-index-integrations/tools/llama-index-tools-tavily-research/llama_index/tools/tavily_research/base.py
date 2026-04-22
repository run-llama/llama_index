"""Tavily tool spec."""

from typing import List, Optional

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class TavilyToolSpec(BaseToolSpec):
    """Tavily tool spec."""

    spec_functions = [
        "search",
        "extract",
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

    def extract(
        self,
        urls: List[str],
        include_images: bool = False,
        include_favicon: bool = False,
        extract_depth: str = "basic",
        format: str = "markdown",
    ) -> List[Document]:
        """
        Extract raw content from a URL using Tavily Extract API.

        Args:
            urls: The URL(s) to extract content from.
            include_images: Whether to include images in the response.
            include_favicon: Whether to include the favicon in the response.
            extract_depth: 'basic' or 'advanced' (default: 'basic').
            format: 'markdown' or 'text' (default: 'markdown').

        Returns:
            A list of Document objects containing the extracted content and metadata,
            or an empty list if no results were returned.

        """
        response = self.client.extract(
            urls,
            include_images=include_images,
            include_favicon=include_favicon,
            extract_depth=extract_depth,
            format=format,
        )

        results = response.get("results", [])

        if not results:
            return []

        return [
            Document(
                text=result.get("raw_content", ""),
                extra_info={
                    "url": result.get("url"),
                    "favicon": result.get("favicon"),
                    "images": result.get("images"),
                },
            )
            for result in results
        ]
