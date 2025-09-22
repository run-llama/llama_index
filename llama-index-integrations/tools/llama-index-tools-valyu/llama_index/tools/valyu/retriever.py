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
