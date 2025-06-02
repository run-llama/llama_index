"""Valyu tool spec."""

from typing import List, Optional

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class ValyuToolSpec(BaseToolSpec):
    """Valyu tool spec."""

    spec_functions = [
        "search",
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

    def search(
        self,
        query: str,
        search_type: str = "all",
        max_num_results: int = 5,
        relevance_threshold: float = 0.5,
        max_price: Optional[float] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Document]:
        """
        Search and retrieve relevant content from proprietary and public sources using Valyu's deep search.

        Args:
            query (str): The input query to be processed
            search_type (str): Type of search - "all" (both proprietary and web), "proprietary" (Valyu indices only), or "web" (web search only). Defaults to "all"
            max_num_results (int): Maximum number of results to return (1-20). Defaults to 5
            relevance_threshold (float): Minimum relevance score required for results (0.0-1.0). Defaults to 0.5
            max_price (Optional[float]): Maximum cost in dollars for this search. Defaults to 20.0
            start_date (Optional[str]): Start date for time filtering in YYYY-MM-DD format
            end_date (Optional[str]): End date for time filtering in YYYY-MM-DD format

        Returns:
            List[Document]: List of Document objects containing the search results

        """
        if max_price is None:
            max_price = self._max_price

        response = self.client.search(
            query=query,
            search_type=search_type,
            max_num_results=max_num_results,
            relevance_threshold=relevance_threshold,
            max_price=max_price,
            start_date=start_date,
            end_date=end_date,
        )

        if self._verbose:
            print(f"[Valyu Tool] Response: {response}")

        documents = []
        for result in response.results:
            metadata = {
                "title": result.title,
                "url": result.url,
                "source": result.source,
                "price": result.price,
                "length": result.length,
                "data_type": result.data_type,
                "relevance_score": result.relevance_score,
            }

            documents.append(
                Document(
                    text=result.content,
                    metadata=metadata,
                )
            )

        return documents
