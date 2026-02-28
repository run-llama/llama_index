"""
Built-Simple ArXiv Reader for LlamaIndex.

This reader connects to Built-Simple's ArXiv Search API, which provides
semantic search over arXiv preprints across physics, mathematics,
computer science, and other scientific disciplines.
"""

from typing import List, Optional
import logging

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from .base import BuiltSimpleBaseClient, BuiltSimpleAPIError, clean_text, format_authors

logger = logging.getLogger(__name__)


class BuiltSimpleArxivReader(BaseReader):
    """
    Reader for Built-Simple's ArXiv Search API.

    This reader provides semantic search access to arXiv's collection of
    scientific preprints. The API uses vector embeddings to find semantically
    similar papers based on your query.

    Features:
        - Semantic vector search over arXiv papers
        - Coverage of physics, math, CS, biology, and more
        - Rich metadata including authors, categories, and links
        - No API key required for basic usage

    Example:
        >>> from llama_index.readers.builtsimple import BuiltSimpleArxivReader
        >>> reader = BuiltSimpleArxivReader()
        >>> documents = reader.load_data("transformer attention mechanisms", limit=10)
        >>> for doc in documents:
        ...     print(f"{doc.metadata['arxiv_id']}: {doc.metadata['title']}")

    """

    BASE_URL = "https://arxiv.built-simple.ai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        """
        Initialize the ArXiv reader.

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
        Search ArXiv and load results as Documents.

        Args:
            query: Search query string (supports natural language)
            limit: Maximum number of results to return (default: 10, max: 100)
            **kwargs: Additional arguments passed to the API

        Returns:
            List of Document objects containing paper abstracts

        Raises:
            BuiltSimpleAPIError: If the API request fails
            ValueError: If query is empty

        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        limit = min(max(1, limit), 100)  # Clamp between 1 and 100

        logger.info(f"Searching ArXiv for: {query} (limit={limit})")

        try:
            response = self.client.get(
                "/api/search",
                params={"q": query.strip(), "limit": limit, **kwargs},
            )
        except BuiltSimpleAPIError as e:
            logger.error(f"ArXiv search failed: {e}")
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

        logger.info(f"Retrieved {len(documents)} documents from ArXiv")
        return documents

    def _result_to_document(self, result: dict, query: str) -> Document:
        """
        Convert an ArXiv result to a LlamaIndex Document.

        Args:
            result: Raw result dictionary from the API
            query: Original search query (for metadata)

        Returns:
            Document object with content and metadata

        """
        arxiv_id = result.get("arxiv_id", result.get("id", ""))
        title = clean_text(result.get("title", ""))
        abstract = clean_text(result.get("abstract", ""))
        authors = format_authors(result.get("authors"))
        year = result.get("year") or result.get("pub_year")
        similarity = result.get("similarity", result.get("score"))
        categories = result.get("categories", [])

        # Build URLs
        abs_url = f"https://arxiv.org/abs/{arxiv_id}"
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        # Build document text
        text_parts = []
        if title:
            text_parts.append(f"Title: {title}")
        if authors:
            text_parts.append(f"Authors: {authors}")
        if abstract:
            text_parts.append(f"\nAbstract: {abstract}")

        text = "\n".join(text_parts) if text_parts else "No content available"

        # Build metadata
        metadata = {
            "source": "arxiv",
            "api": "built-simple",
            "arxiv_id": arxiv_id,
            "title": title,
            "url": abs_url,
            "pdf_url": pdf_url,
            "query": query,
        }

        # Add optional metadata
        if authors:
            metadata["authors"] = authors
        if year:
            metadata["year"] = int(year) if str(year).isdigit() else year
        if similarity is not None:
            metadata["similarity_score"] = float(similarity)
        if categories:
            if isinstance(categories, list):
                metadata["categories"] = ", ".join(categories)
            else:
                metadata["categories"] = str(categories)

        # Add additional fields if present
        if "doi" in result:
            metadata["doi"] = result["doi"]
        if "journal_ref" in result:
            metadata["journal_ref"] = result["journal_ref"]
        if "primary_category" in result:
            metadata["primary_category"] = result["primary_category"]

        return Document(
            text=text,
            metadata=metadata,
            doc_id=f"arxiv_{arxiv_id.replace('.', '_').replace('/', '_')}",
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
