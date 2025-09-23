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
        # Search API parameters
        max_price: Optional[float] = 100,
        relevance_threshold: float = 0.5,
        fast_mode: Optional[bool] = False,
        included_sources: Optional[List[str]] = None,
        excluded_sources: Optional[List[str]] = None,
        response_length: Optional[Union[int, str]] = None,
        country_code: Optional[str] = None,
        # Contents API parameters
        contents_summary: Optional[Union[bool, str, Dict[str, Any]]] = None,
        contents_extract_effort: Optional[str] = "normal",
        contents_response_length: Optional[Union[str, int]] = "short",
    ) -> None:
        """
        Initialize with parameters.

        Args:
            api_key (str): Valyu API key
            verbose (bool): Enable verbose logging. Defaults to False
            max_price (Optional[float]): Maximum cost in dollars for search operations. Defaults to 100
            relevance_threshold (float): Minimum relevance score required for results (0.0-1.0). Defaults to 0.5
            fast_mode (Optional[bool]): Enable fast mode for faster but shorter results. If None, model can decide per search (defaults to False if not specified). Defaults to None
            included_sources (Optional[List[str]]): List of URLs, domains or datasets to only search over and return in results. Defaults to None
            excluded_sources (Optional[List[str]]): List of URLs, domains or datasets to exclude from search results. Defaults to None
            response_length (Optional[Union[int, str]]): Number of characters to return per item or preset values: "short" (25k chars), "medium" (50k chars), "large" (100k chars), "max" (full content). Defaults to None
            country_code (Optional[str]): 2-letter ISO country code (e.g., "GB", "US") to bias search results to a specific country. Defaults to None
            contents_summary (Optional[Union[bool, str, Dict[str, Any]]]): AI summary configuration:
                - False/None: No AI processing (raw content)
                - True: Basic automatic summarization
                - str: Custom instructions (max 500 chars)
                - dict: JSON schema for structured extraction
            contents_extract_effort (Optional[str]): Extraction thoroughness:
                - "normal": Fast extraction (default)
                - "high": More thorough but slower
                - "auto": Automatically determine extraction effort but slowest
            contents_response_length (Optional[Union[str, int]]): Content length per URL:
                - "short": 25,000 characters (default)
                - "medium": 50,000 characters
                - "large": 100,000 characters
                - "max": No limit
                - int: Custom character limit

        """
        from valyu import Valyu

        # Validate parameters
        if not api_key or not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key must be a non-empty string")

        if max_price is not None and (
            not isinstance(max_price, (int, float)) or max_price < 0
        ):
            raise ValueError("max_price must be a non-negative number or None")

        if (
            not isinstance(relevance_threshold, (int, float))
            or relevance_threshold < 0.0
            or relevance_threshold > 1.0
        ):
            raise ValueError("relevance_threshold must be a number between 0.0 and 1.0")

        if not isinstance(verbose, bool):
            raise ValueError("verbose must be a boolean")

        if fast_mode is not None and not isinstance(fast_mode, bool):
            raise ValueError("fast_mode must be a boolean or None")

        # Validate search parameters
        if included_sources is not None:
            if not isinstance(included_sources, list) or not all(
                isinstance(s, str) for s in included_sources
            ):
                raise ValueError("included_sources must be a list of strings or None")

        if excluded_sources is not None:
            if not isinstance(excluded_sources, list) or not all(
                isinstance(s, str) for s in excluded_sources
            ):
                raise ValueError("excluded_sources must be a list of strings or None")

        # Validate response_length
        if response_length is not None:
            valid_preset_lengths = ["short", "medium", "large", "max"]
            if isinstance(response_length, str):
                if response_length not in valid_preset_lengths:
                    raise ValueError(
                        f"response_length string must be one of {valid_preset_lengths}"
                    )
            elif isinstance(response_length, int):
                if response_length < 1:
                    raise ValueError(
                        "response_length must be a positive integer when using custom length"
                    )
            else:
                raise ValueError(
                    "response_length must be a string preset, positive integer, or None"
                )

        # Validate country_code
        if country_code is not None:
            if (
                not isinstance(country_code, str)
                or len(country_code) != 2
                or not country_code.isalpha()
            ):
                raise ValueError(
                    "country_code must be a 2-letter ISO country code (e.g., 'GB', 'US') or None"
                )

        # Validate contents_summary
        if contents_summary is not None:
            if isinstance(contents_summary, str):
                if len(contents_summary) > 500:
                    raise ValueError(
                        f"contents_summary string must be 500 characters or less. "
                        f"Current length: {len(contents_summary)} characters."
                    )
            elif not isinstance(contents_summary, (bool, dict)):
                raise ValueError(
                    "contents_summary must be a boolean, string, dict, or None"
                )

        # Validate contents_extract_effort
        valid_extract_efforts = ["normal", "high", "auto"]
        if (
            contents_extract_effort is not None
            and contents_extract_effort not in valid_extract_efforts
        ):
            raise ValueError(
                f"contents_extract_effort must be one of {valid_extract_efforts}"
            )

        # Validate contents_response_length
        if contents_response_length is not None:
            valid_preset_lengths = ["short", "medium", "large", "max"]
            if isinstance(contents_response_length, str):
                if contents_response_length not in valid_preset_lengths:
                    raise ValueError(
                        f"contents_response_length string must be one of {valid_preset_lengths}"
                    )
            elif isinstance(contents_response_length, int):
                if contents_response_length < 1:
                    raise ValueError(
                        "contents_response_length must be a positive integer when using custom length"
                    )
            else:
                raise ValueError(
                    "contents_response_length must be a string preset, positive integer, or None"
                )

        self.client = Valyu(api_key=api_key)
        self._verbose = verbose
        self._max_price = max_price
        self._relevance_threshold = relevance_threshold
        self._fast_mode = fast_mode
        # Search API defaults
        self._included_sources = included_sources
        self._excluded_sources = excluded_sources
        self._response_length = response_length
        self._country_code = country_code
        # Contents API defaults
        self._contents_summary = contents_summary
        self._contents_extract_effort = contents_extract_effort
        self._contents_response_length = contents_response_length

    def search(
        self,
        query: str,
        search_type: str = "all",
        max_num_results: int = 5,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        fast_mode: Optional[bool] = None,
    ) -> List[Document]:
        """
        Search and retrieve relevant content from proprietary and public sources using Valyu's deep search.

        Args:
            query (str): The input query to be processed
            search_type (str): Type of search - "all" (both proprietary and web), "proprietary" (Valyu indices only), or "web" (web search only). Defaults to "all"
            max_num_results (int): Maximum number of results to return (1-20). Defaults to 5
            start_date (Optional[str]): Start date for time filtering in YYYY-MM-DD format
            end_date (Optional[str]): End date for time filtering in YYYY-MM-DD format
            fast_mode (Optional[bool]): Enable fast mode for faster but shorter results. If None, uses the default set during initialization, or defaults to False if no user default was set

        Returns:
            List[Document]: List of Document objects containing the search results

        Note:
            The following parameters are set during tool initialization and cannot be modified per search:
            - max_price: Maximum cost limit for search operations
            - relevance_threshold: Minimum relevance score required for results
            - included_sources: List of sources to include in search
            - excluded_sources: List of sources to exclude from search
            - response_length: Response length configuration
            - country_code: Country bias for search results

        """
        # Handle fast_mode: if user set a specific value (not None), always use it
        # If user set None, then model can decide, but we need to provide a boolean to the SDK
        if self._fast_mode is not None:
            fast_mode = self._fast_mode  # User controls fast_mode
        elif fast_mode is None:
            # Both user and model didn't specify, use SDK default (False)
            fast_mode = False

        response = self.client.search(
            query=query,
            search_type=search_type,
            max_num_results=max_num_results,
            relevance_threshold=self._relevance_threshold,
            max_price=self._max_price,
            start_date=start_date,
            end_date=end_date,
            included_sources=self._included_sources,
            excluded_sources=self._excluded_sources,
            response_length=self._response_length,
            country_code=self._country_code,
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
