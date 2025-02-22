"""Valyu tool spec."""

from typing import List, Optional

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class ValyuToolSpec(BaseToolSpec):
    """Valyu tool spec."""

    spec_functions = [
        "context",
    ]

    def __init__(
        self,
        api_key: str,
        verbose: bool = False,
        max_price: Optional[float] = 100,
    ) -> None:
        """Initialize with parameters."""
        from valyu import Valyu

        self.client = Valyu(api_key=api_key)
        self._verbose = verbose
        self._max_price = max_price

    def context(
        self,
        query: str,
        search_type: str = "both",
        data_sources: Optional[List[str]] = None,
        max_num_results: int = 10,
        max_price: Optional[float] = None,
        query_rewrite: bool = True,
        similarity_threshold: float = 0.5,
    ) -> List[Document]:
        """Find relevant programmaticly licensed proprietary content and the web to answer your query.

        Args:
            query (str): The question or topic to search for
            search_type (str): Type of sources to search - "proprietary", "web", or "both"
            data_sources (Optional[List[str]]): Specific indexes to query from
            max_num_results (int): Maximum number of results to return. Defaults to 10
            max_price (Optional[float]): Maximum price per content in PCM. Defaults to 100
            query_rewrite (bool): Whether to rewrite the query to improve results. Defaults to True
            similarity_threshold (float): Minimum similarity score for results. Defaults to 0.5

        Returns:
            List[Document]: List of Document objects containing the search results
        """
        if max_price is None:
            max_price = self._max_price

        response = self.client.context(
            query=query,
            search_type=search_type,
            data_sources=data_sources,
            max_num_results=max_num_results,
            max_price=max_price,
            query_rewrite=query_rewrite,
            similarity_threshold=similarity_threshold,
        )

        if self._verbose:
            print(f"[Valyu Tool] Response: {response}")

        return [
            Document(
                text=result.content,
                metadata={
                    "title": result.title,
                    "url": result.url,
                    "source": result.source,
                    "price": result.price,
                },
            )
            for result in response.results
        ]
