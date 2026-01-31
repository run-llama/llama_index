"""
Built-Simple Wikipedia Reader for LlamaIndex.

This reader connects to Built-Simple's Wikipedia Search API, which provides
semantic search over Wikipedia articles.
"""

from typing import List, Optional
import logging

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from .base import BuiltSimpleBaseClient, BuiltSimpleAPIError, clean_text

logger = logging.getLogger(__name__)


class BuiltSimpleWikipediaReader(BaseReader):
    """
    Reader for Built-Simple's Wikipedia Search API.

    This reader provides semantic search access to Wikipedia articles.
    The API uses vector embeddings to find semantically similar articles
    based on your query.

    Note:
        This service may have limited availability. For production use,
        consider implementing fallback logic.

    Features:
        - Semantic vector search over Wikipedia
        - Rich metadata including page IDs and URLs
        - No API key required for basic usage

    Example:
        >>> from llama_index.readers.builtsimple import BuiltSimpleWikipediaReader
        >>> reader = BuiltSimpleWikipediaReader()
        >>> documents = reader.load_data("quantum computing applications", limit=5)
        >>> for doc in documents:
        ...     print(doc.metadata["title"])

    """

    BASE_URL = "https://wikipedia.built-simple.ai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        """
        Initialize the Wikipedia reader.

        Args:
            api_key: Optional API key for higher rate limits.
                    Free tier available without key.
            timeout: Request timeout in seconds (default: 30)

        """
        super().__init__()
        self.client = BuiltSimpleBaseClient(
            base_url=self.BASE_URL,
            api_key=api_key,
            timeout=timeout,
        )

    def load_data(
        self,
        query: str,
        limit: int = 10,
        **kwargs,
    ) -> List[Document]:
        """
        Search Wikipedia and load results as Documents.

        Args:
            query: Search query string (supports natural language)
            limit: Maximum number of results to return (default: 10, max: 100)
            **kwargs: Additional arguments passed to the API

        Returns:
            List of Document objects containing article content

        Raises:
            BuiltSimpleAPIError: If the API request fails
            ValueError: If query is empty

        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        limit = min(max(1, limit), 100)  # Clamp between 1 and 100

        logger.info(f"Searching Wikipedia for: {query} (limit={limit})")

        try:
            # Try POST first (hybrid-search endpoint)
            response = self.client.post(
                "/hybrid-search",
                data={"query": query.strip(), "limit": limit, **kwargs},
            )
        except BuiltSimpleAPIError:
            try:
                # Fallback to GET endpoint
                response = self.client.get(
                    "/api/search",
                    params={"q": query.strip(), "limit": limit, **kwargs},
                )
            except BuiltSimpleAPIError as e:
                logger.error(f"Wikipedia search failed: {e}")
                raise

        results = response.get("results", [])
        documents = []

        for result in results:
            try:
                doc = self._result_to_document(result, query)
                documents.append(doc)
            except Exception as e:
                logger.warning(f"Failed to parse result: {e}")
                continue

        logger.info(f"Retrieved {len(documents)} documents from Wikipedia")
        return documents

    def _result_to_document(self, result: dict, query: str) -> Document:
        """
        Convert a Wikipedia result to a LlamaIndex Document.

        Args:
            result: Raw result dictionary from the API
            query: Original search query (for metadata)

        Returns:
            Document object with content and metadata

        """
        # Handle various possible field names
        page_id = result.get("page_id", result.get("id", result.get("pageid", "")))
        title = clean_text(result.get("title", ""))
        content = clean_text(
            result.get("content")
            or result.get("text")
            or result.get("extract")
            or result.get("summary", "")
        )
        url = result.get("url") or (
            f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}" if title else ""
        )
        similarity = result.get("similarity", result.get("score"))

        # Build document text
        text_parts = []
        if title:
            text_parts.append(f"Title: {title}")
        if content:
            text_parts.append(f"\n{content}")

        text = "\n".join(text_parts) if text_parts else "No content available"

        # Build metadata
        metadata = {
            "source": "wikipedia",
            "api": "built-simple",
            "title": title,
            "url": url,
            "query": query,
        }

        # Add optional metadata
        if page_id:
            metadata["page_id"] = str(page_id)
        if similarity is not None:
            metadata["similarity_score"] = float(similarity)

        # Add additional fields if present
        if "categories" in result:
            categories = result["categories"]
            if isinstance(categories, list):
                metadata["categories"] = ", ".join(categories)
            else:
                metadata["categories"] = str(categories)
        if "last_modified" in result:
            metadata["last_modified"] = result["last_modified"]
        if "language" in result:
            metadata["language"] = result["language"]

        # Generate document ID
        doc_id = (
            f"wikipedia_{page_id}"
            if page_id
            else f"wikipedia_{title.replace(' ', '_')}"
        )

        return Document(
            text=text,
            metadata=metadata,
            doc_id=doc_id,
        )

    def lazy_load_data(self, query: str, limit: int = 10, **kwargs):
        """
        Lazily load documents (yields one at a time).

        This is more memory-efficient for large result sets.

        Args:
            query: Search query string
            limit: Maximum number of results
            **kwargs: Additional arguments

        Yields:
            Document objects one at a time

        """
        documents = self.load_data(query, limit, **kwargs)
        for doc in documents:
            yield doc
