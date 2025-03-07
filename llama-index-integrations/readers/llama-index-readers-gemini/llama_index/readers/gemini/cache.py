"""Caching implementation for Gemini PDF Reader."""

import hashlib
import json
import logging
import os
import time
from typing import Dict, List, Optional, Any

from llama_index.core.schema import Document
from pydantic import BaseModel

from llama_index.readers.gemini.types import ProcessingStats

# Configure logging
logger = logging.getLogger(__name__)


class CacheItem(BaseModel):
    """Cache item for storing processed PDF results."""

    timestamp: float
    document_hash: str
    documents: List[Dict[str, Any]]
    stats: ProcessingStats


class CacheManager:
    """Manages caching of processed PDF documents."""

    def __init__(
        self, enable_caching: bool, cache_dir: str,
        cache_ttl: int, verbose: bool
    ):
        """
        Initialize the cache manager.

        Args:
            enable_caching: Whether caching is enabled
            cache_dir: Directory for cache storage
            cache_ttl: Time-to-live for cache entries in seconds
            verbose: Whether to log verbose messages
        """
        self.enable_caching = enable_caching
        self.cache_dir = cache_dir
        self.cache_ttl = cache_ttl
        self.verbose = verbose
        self._cache: Dict[str, CacheItem] = {}

        if self.enable_caching:
            self._initialize_cache()

    def _initialize_cache(self) -> None:
        """Initialize the cache directory and load existing cache entries."""
        if not self.enable_caching:
            return

        print(type(self.cache_dir))
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load existing cache entries
        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if filename.endswith(".json"):
                    try:
                        with open(os.path.join(self.cache_dir, filename)) as f:
                            cache_data = json.load(f)
                            cache_item = CacheItem(**cache_data)

                            # Check if the cache entry is still valid
                            if (
                             time.time() - cache_item.timestamp) < (
                             self.cache_ttl):
                                self._cache[
                                    cache_item.document_hash] = cache_item
                    except Exception as e:
                        if self.verbose:
                            logger.warning(
                                f"Failed to load cache entry {filename}: {e!s}"
                            )

    def compute_file_hash(self, file_path: str) -> str:
        """
        Compute a hash of the file for caching purposes.

        Args:
            file_path: Path to the file

        Returns:
            MD5 hash string
        """
        file_stat = os.stat(file_path)
        # Use file path, size and modification time for the hash
        hash_string = f"{file_path}:{file_stat.st_size}:{file_stat.st_mtime}"
        return hashlib.md5(hash_string.encode()).hexdigest()

    def save_to_cache(
        self, file_hash: str, documents: List[Document], stats: ProcessingStats
    ) -> None:
        """
        Save processed results to cache.

        Args:
            file_hash: Hash identifier for the document
            documents: Processed document chunks
            stats: Processing statistics
        """
        if not self.enable_caching:
            return

        try:
            # Convert Documents to dictionaries
            doc_dicts = []
            for doc in documents:
                doc_dict = {"text": doc.text, "metadata": doc.metadata}
                doc_dicts.append(doc_dict)

            # Create cache item
            cache_item = CacheItem(
                timestamp=time.time(),
                document_hash=file_hash,
                documents=doc_dicts,
                stats=stats,
            )

            # Save to memory cache
            self._cache[file_hash] = cache_item

            # Save to disk cache
            cache_path = os.path.join(self.cache_dir, f"{file_hash}.json")
            with open(cache_path, "w") as f:
                json.dump(cache_item.model_dump(), f)

            if self.verbose:
                logger.info(
                    f"Cached results for {file_hash} (documents: {len(
                        documents)})"
                )

        except Exception as e:
            logger.warning(f"Failed to cache results: {e!s}")

    def load_from_cache(self, file_hash: str) -> Optional[List[Document]]:
        """
        Load processed results from cache if available.

        Args:
            file_hash: Hash identifier for the document

        Returns:
            Cached document chunks or None if not in cache
        """
        if not self.enable_caching:
            return None

        cache_item = self._cache.get(file_hash)
        if not cache_item:
            return None

        # Check if the cache entry is still valid
        if time.time() - cache_item.timestamp > self.cache_ttl:
            if self.verbose:
                logger.info(f"Cache entry expired for {file_hash}")
            return None

        # Convert cached dictionaries back to Documents
        documents = []
        for doc_dict in cache_item.documents:
            doc = Document(
                text=doc_dict["text"], metadata=doc_dict["metadata"])
            documents.append(doc)

        if self.verbose:
            logger.info(
             f"Loaded {len(documents)} documents from cache for {file_hash}")

        return documents
