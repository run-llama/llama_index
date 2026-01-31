"""
Built-Simple PubMed Reader for LlamaIndex.

This reader connects to Built-Simple's PubMed Hybrid Search API, which provides
semantic and keyword search over the PubMed biomedical literature database.
"""

from typing import List, Optional
import logging

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from .base import BuiltSimpleBaseClient, BuiltSimpleAPIError, clean_text, format_authors

logger = logging.getLogger(__name__)


class BuiltSimplePubMedReader(BaseReader):
    """
    Reader for Built-Simple's PubMed Hybrid Search API.

    This reader provides access to PubMed's vast collection of biomedical
    literature through Built-Simple's hybrid search endpoint, which combines
    semantic understanding with traditional keyword matching for optimal results.

    Features:
        - Hybrid search combining BM25 and vector similarity
        - Access to 4.5M+ PubMed articles with FULL TEXT available
        - Rich metadata including journal, authors, DOI
        - No API key required for basic usage

    Example:
        >>> from llama_index.readers.builtsimple import BuiltSimplePubMedReader
        >>> reader = BuiltSimplePubMedReader()
        >>> # Abstract only (default)
        >>> documents = reader.load_data("CRISPR gene therapy", limit=10)
        >>> # With full text
        >>> documents = reader.load_data("CRISPR gene therapy", limit=10, include_full_text=True)
        >>> for doc in documents:
        ...     print(doc.metadata["title"])

    """

    BASE_URL = "https://pubmed.built-simple.ai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 30,
        include_full_text: bool = False,
    ):
        """
        Initialize the PubMed reader.

        Args:
            api_key: Optional API key for higher rate limits.
                    Free tier available without key.
            timeout: Request timeout in seconds (default: 30)
            include_full_text: Whether to fetch full article text by default.
                              When True, makes additional API calls for each result.
                              (default: False, abstracts only)

        """
        super().__init__()
        self.client = BuiltSimpleBaseClient(
            base_url=self.BASE_URL,
            api_key=api_key,
            timeout=timeout,
        )
        self.include_full_text = include_full_text

    def load_data(
        self,
        query: str,
        limit: int = 10,
        include_full_text: Optional[bool] = None,
        **kwargs,
    ) -> List[Document]:
        """
        Search PubMed and load results as Documents.

        Args:
            query: Search query string (supports natural language)
            limit: Maximum number of results to return (default: 10, max: 100)
            include_full_text: Override default full text setting.
                              When True, fetches full article text (not just abstracts).
                              Makes additional API calls for each result.
            **kwargs: Additional arguments passed to the API

        Returns:
            List of Document objects containing article text

        Raises:
            BuiltSimpleAPIError: If the API request fails
            ValueError: If query is empty

        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Use instance default if not specified
        fetch_full_text = (
            include_full_text
            if include_full_text is not None
            else self.include_full_text
        )

        limit = min(max(1, limit), 100)  # Clamp between 1 and 100

        logger.info(
            f"Searching PubMed for: {query} (limit={limit}, full_text={fetch_full_text})"
        )

        try:
            response = self.client.post(
                "/hybrid-search",
                data={"query": query.strip(), "limit": limit, **kwargs},
            )
        except BuiltSimpleAPIError as e:
            logger.error(f"PubMed search failed: {e}")
            raise

        results = response.get("results", [])
        documents = []

        for result in results:
            try:
                # Optionally fetch full text
                full_text_data = None
                if fetch_full_text:
                    pmid = result.get("pmid", "")
                    if pmid:
                        full_text_data = self._fetch_full_text(pmid)

                doc = self._result_to_document(result, query, full_text_data)
                documents.append(doc)
            except Exception as e:
                logger.warning(f"Failed to parse result: {e}")
                continue

        logger.info(f"Retrieved {len(documents)} documents from PubMed")
        return documents

    def _fetch_full_text(self, pmid: str) -> Optional[dict]:
        """
        Fetch full article text for a given PMID.

        Args:
            pmid: PubMed ID

        Returns:
            Dictionary with full text data, or None if unavailable

        """
        try:
            # Clean PMID (remove PMC prefix if present for the API call)
            clean_pmid = pmid.replace("PMC", "")
            response = self.client.get(f"/article/{pmid}/full_text")
            if response.get("has_full_text") or response.get("full_text"):
                return response
        except Exception as e:
            logger.debug(f"Full text not available for {pmid}: {e}")
        return None

    def _result_to_document(
        self, result: dict, query: str, full_text_data: Optional[dict] = None
    ) -> Document:
        """
        Convert a PubMed result to a LlamaIndex Document.

        Args:
            result: Raw result dictionary from the API
            query: Original search query (for metadata)
            full_text_data: Optional full text response from the API

        Returns:
            Document object with content and metadata

        """
        pmid = result.get("pmid", "")
        title = clean_text(result.get("title", ""))
        abstract = clean_text(result.get("abstract", ""))
        journal = result.get("journal", "")
        pub_year = result.get("pub_year") or result.get("year")
        doi = result.get("doi", "")
        url = result.get("url") or f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        authors = format_authors(result.get("authors"))

        # Use full text if available, otherwise fall back to abstract
        has_full_text = False
        if full_text_data and full_text_data.get("full_text"):
            full_text = clean_text(full_text_data["full_text"])
            has_full_text = True
            # Build document text with full article
            text_parts = [f"Title: {title}"]
            if authors:
                text_parts.append(f"Authors: {authors}")
            text_parts.append(f"\nFull Text:\n{full_text}")
        else:
            # Build document text with abstract only
            text_parts = []
            if title:
                text_parts.append(f"Title: {title}")
            if abstract:
                text_parts.append(f"\nAbstract: {abstract}")

        text = "\n".join(text_parts) if text_parts else "No content available"

        # Build metadata
        metadata = {
            "source": "pubmed",
            "api": "built-simple",
            "pmid": pmid,
            "title": title,
            "journal": journal,
            "url": url,
            "query": query,
            "has_full_text": has_full_text,
        }

        # Add optional metadata
        if pub_year:
            metadata["year"] = int(pub_year) if str(pub_year).isdigit() else pub_year
        if doi:
            metadata["doi"] = doi
            metadata["doi_url"] = f"https://doi.org/{doi}"
        if authors:
            metadata["authors"] = authors
        if full_text_data:
            metadata["full_text_length"] = full_text_data.get(
                "full_text_length", len(full_text_data.get("full_text", ""))
            )

        # Add similarity score if present
        if "similarity" in result:
            metadata["similarity_score"] = result["similarity"]
        if "score" in result:
            metadata["score"] = result["score"]

        return Document(
            text=text,
            metadata=metadata,
            doc_id=f"pubmed_{pmid}",
        )

    def load_full_text(self, pmid: str) -> Optional[Document]:
        """
        Load a single article by PMID with full text.

        Args:
            pmid: PubMed ID (e.g., "31041627" or "PMC9953887")

        Returns:
            Document with full article text, or None if not found

        """
        full_text_data = self._fetch_full_text(pmid)
        if not full_text_data:
            logger.warning(f"No full text available for PMID {pmid}")
            return None

        # Create a result-like dict from full text response
        result = {
            "pmid": pmid,
            "title": full_text_data.get("title", ""),
            "abstract": full_text_data.get("abstract", ""),
            "journal": full_text_data.get("journal", ""),
            "pub_year": full_text_data.get("pub_year"),
            "doi": full_text_data.get("doi", ""),
        }

        return self._result_to_document(
            result, query="direct_lookup", full_text_data=full_text_data
        )

    def lazy_load_data(
        self,
        query: str,
        limit: int = 10,
        include_full_text: Optional[bool] = None,
        **kwargs,
    ):
        """
        Lazily load documents (yields one at a time).

        This is more memory-efficient for large result sets.

        Args:
            query: Search query string
            limit: Maximum number of results
            include_full_text: Whether to fetch full text
            **kwargs: Additional arguments

        Yields:
            Document objects one at a time

        """
        documents = self.load_data(
            query, limit, include_full_text=include_full_text, **kwargs
        )
        for doc in documents:
            yield doc
