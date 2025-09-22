"""Valyu tool spec."""

from typing import List, Optional, Union, Dict, Any

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class ValyuToolSpec(BaseToolSpec):
    """Valyu tool spec."""

    spec_functions = [
        "search",
        "get_contents",
    ]

    def __init__(
        self,
        api_key: str,
        verbose: bool = False,
        max_price: Optional[float] = 100,
        fast_mode: bool = False,
        # Contents API parameters
        contents_summary: Optional[Union[bool, str, Dict[str, Any]]] = None,
        contents_extract_effort: Optional[str] = "normal",
        contents_response_length: Optional[Union[str, int]] = "short",
    ) -> None:
        """Initialize with parameters."""
        from valyu import Valyu

        self.client = Valyu(api_key=api_key)
        self._verbose = verbose
        self._max_price = max_price
        self._fast_mode = fast_mode
        # Contents API defaults
        self._contents_summary = contents_summary
        self._contents_extract_effort = contents_extract_effort
        self._contents_response_length = contents_response_length

    def search(
        self,
        query: str,
        search_type: str = "all",
        max_num_results: int = 5,
        relevance_threshold: float = 0.5,
        max_price: Optional[float] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        included_sources: Optional[List[str]] = None,
        excluded_sources: Optional[List[str]] = None,
        response_length: Optional[Union[int, str]] = None,
        country_code: Optional[str] = None,
        fast_mode: Optional[bool] = None,
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
            included_sources (Optional[List[str]]): List of URLs, domains or datasets to only search over and return in results
            excluded_sources (Optional[List[str]]): List of URLs, domains or datasets to exclude from search results
            response_length (Optional[Union[int, str]]): Number of characters to return per item or preset values: "short" (25k chars), "medium" (50k chars), "large" (100k chars), "max" (full content)
            country_code (Optional[str]): 2-letter ISO country code (e.g., "GB", "US") to bias search results to a specific country
            fast_mode (Optional[bool]): Enable fast mode for faster but shorter results. Good for general purpose queries. If None, uses the default set during initialization. Defaults to None

        Returns:
            List[Document]: List of Document objects containing the search results

        """
        if max_price is None:
            max_price = self._max_price

        if fast_mode is None:
            fast_mode = self._fast_mode

        response = self.client.search(
            query=query,
            search_type=search_type,
            max_num_results=max_num_results,
            relevance_threshold=relevance_threshold,
            max_price=max_price,
            start_date=start_date,
            end_date=end_date,
            included_sources=included_sources,
            excluded_sources=excluded_sources,
            response_length=response_length,
            country_code=country_code,
            fast_mode=fast_mode,
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

    def get_contents(
        self,
        urls: List[str],
    ) -> List[Document]:
        """
        Extract clean, structured content from web pages.

        This method fetches the content from the provided URLs using Valyu's content extraction API.
        The extraction parameters (summary, extract_effort, response_length, max_price) are set
        during tool initialization and cannot be modified by the model - only the URLs can be specified.

        Args:
            urls (List[str]): List of URLs to extract content from (maximum 10 URLs per request)

        Returns:
            List[Document]: List of Document objects containing the extracted content

        """
        response = self.client.contents(
            urls=urls,
            summary=self._contents_summary,
            extract_effort=self._contents_extract_effort,
            response_length=self._contents_response_length,
        )

        if self._verbose:
            print(f"[Valyu Tool] Contents Response: {response}")

        documents = []
        if response and response.results:
            for result in response.results:
                metadata = {
                    "url": result.url,
                    "title": result.title,
                    "source": result.source,
                    "length": result.length,
                    "data_type": result.data_type,
                    "citation": result.citation,
                }

                # Add summary info if available
                if hasattr(result, "summary") and result.summary:
                    metadata["summary"] = result.summary
                if (
                    hasattr(result, "summary_success")
                    and result.summary_success is not None
                ):
                    metadata["summary_success"] = result.summary_success

                # Add image URL if available
                if hasattr(result, "image_url") and result.image_url:
                    metadata["image_url"] = result.image_url

                documents.append(
                    Document(
                        text=str(result.content),
                        metadata=metadata,
                    )
                )

        return documents
