"""Valyu retriever implementation."""

from typing import List, Optional, Union, Dict, Any

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.callbacks import CallbackManager


class ValyuRetriever(BaseRetriever):
    """Valyu retriever for extracting content from URLs."""

    def __init__(
        self,
        api_key: str,
        verbose: bool = False,
        # Contents API parameters
        contents_summary: Optional[Union[bool, str, Dict[str, Any]]] = None,
        contents_extract_effort: Optional[str] = "normal",
        contents_response_length: Optional[Union[str, int]] = "short",
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        """
        Initialize Valyu retriever.

        Args:
            api_key (str): Valyu API key
            verbose (bool): Enable verbose logging. Defaults to False
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
            callback_manager (Optional[CallbackManager]): Callback manager for tracking operations

        """
        from valyu import Valyu

        # Validate parameters
        if not api_key or not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key must be a non-empty string")

        if not isinstance(verbose, bool):
            raise ValueError("verbose must be a boolean")

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
        self._contents_summary = contents_summary
        self._contents_extract_effort = contents_extract_effort
        self._contents_response_length = contents_response_length

        super().__init__(callback_manager=callback_manager)

    def _retrieve(self, query_bundle) -> List[NodeWithScore]:
        """
        Retrieve content from URLs.

        The query_bundle.query_str should contain URLs (space or comma separated).
        This method extracts content from those URLs and returns them as scored nodes.

        Args:
            query_bundle: Query bundle containing URLs to extract content from

        Returns:
            List[NodeWithScore]: List of nodes with extracted content and relevance scores

        """
        # Parse URLs from query string
        urls = self._parse_urls_from_query(query_bundle.query_str)

        if not urls:
            return []

        # Get content using Valyu API
        response = self.client.contents(
            urls=urls,
            summary=self._contents_summary,
            extract_effort=self._contents_extract_effort,
            response_length=self._contents_response_length,
        )

        if self._verbose:
            print(f"[Valyu Retriever] Contents Response: {response}")

        nodes = []
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

                # Create text node
                node = TextNode(
                    text=str(result.content),
                    metadata=metadata,
                )

                # Add as scored node (all retrieved content gets score of 1.0)
                nodes.append(NodeWithScore(node=node, score=1.0))

        return nodes

    def _parse_urls_from_query(self, query_str: str) -> List[str]:
        """
        Parse URLs from query string.

        Args:
            query_str: String containing URLs (space or comma separated)

        Returns:
            List[str]: List of valid URLs

        """
        # Split by common separators
        import re

        # Split by whitespace or commas
        potential_urls = re.split(r"[,\s]+", query_str.strip())

        # Filter for valid URLs
        urls = []
        for url in potential_urls:
            url = url.strip()
            if url and url.startswith(("http://", "https://")):
                urls.append(url)

        return urls[:10]  # Limit to 10 URLs as per API constraint
