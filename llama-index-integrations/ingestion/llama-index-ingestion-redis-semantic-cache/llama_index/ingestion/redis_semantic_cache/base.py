from typing import Any, Dict, List, Optional
import json
import struct
import logging

from redis import Redis
from redisvl.extensions.cache.llm import SemanticCache

from llama_index.core.bridge.pydantic import (
    BaseModel,
    Field,
)
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.ingestion.redis_semantic_cache.vectorizer import LlamaIndexVectorizer


DEFAULT_CACHE_NAME = "semantic_cache"
DEFAULT_CACHE_PREFIX = "llama_index"
DEFAULT_CACHE_TTL = 3600
DEFAULT_CACHE_DISTANCE_THRESHOLD = 0.2
LOGGER = logging.getLogger(__name__)


class CacheResult(BaseModel):
    id: str = Field(..., description="unique identifier for this cache entry.")
    key: str = Field(..., description="full Redis key for this cache entry.")
    query: str = Field(..., description="cached query text.")
    response: str = Field(..., description="cached response text.")
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="optional metadata associated with this cache entry."
    )
    vector_distance: float = Field(
        ..., description="semantic distance from vector index (lower = more similar)."
    )
    cosine_similarity: float = Field(
        ..., description="cosine similarity score (higher = more similar)."
    )
    inserted_at: float = Field(
        ..., description="timestamp when this cache entry was created."
    )
    updated_at: float = Field(
        ..., description="timestamp when this cache entry was last updated."
    )


class CacheResults(BaseModel):
    matches: List[CacheResult]


class RedisSemanticCache:
    def __init__(
        self,
        embed_model: BaseEmbedding,
        embedding_dims: int,
        redis_url: str,
        distance_threshold: float = DEFAULT_CACHE_DISTANCE_THRESHOLD,
        ttl: Optional[int] = DEFAULT_CACHE_TTL,
        name: Optional[str] = DEFAULT_CACHE_NAME,
        prefix: Optional[str] = DEFAULT_CACHE_PREFIX,
    ):
        self.embed_model = embed_model
        self.redis_url = redis_url
        self.distance_threshold = distance_threshold
        self.ttl = ttl
        self.name = name
        self.prefix = prefix
        self.vectorizer = LlamaIndexVectorizer(
            embed_model=embed_model,
            embedding_dims=embedding_dims,
        )

        if redis_url.startswith("redis+sentinel://"):
            # For Sentinel URLs, use RedisVL's connection factory
            from redisvl.redis.connection import RedisConnectionFactory

            self.redis_client = RedisConnectionFactory.get_redis_connection(
                redis_url=redis_url
            )
        else:
            self.redis_client = Redis.from_url(redis_url)
            self.cache = SemanticCache(
                name=self.name,
                distance_threshold=self.distance_threshold,
                ttl=self.ttl,
                vectorizer=self.vectorizer,
                redis_client=self.redis_client,
                redis_url=self.redis_url,
                overwrite=False,
            )

    def __len__(self) -> int:
        """Return the number of entries currently in the cache."""
        return self.redis_client.ft(self.name).info().get("num_docs", 0)

    def store_cache_entry(
        self,
        query: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> None:
        """
        Store a query-response pair in the semantic cache with optional metadata and filters.

        Args:
            query: The query string to cache.
            response: The response string to cache.
            metadata: Optional dictionary of metadata to associate with this cache entry.
            filters: Optional dictionary of filters to apply when retrieving this entry.
            ttl: Optional time-to-live for this cache entry in seconds (overrides default TTL if provided).

        Returns:
            None

        """
        try:
            key = self.cache.store(
                prompt=query,
                response=response,
                metadata=metadata,
                filters=filters,
                ttl=ttl if ttl else self.ttl,
            )
            LOGGER.info(f"Successfully stored cache entry with key: {key}")
        except Exception as e:
            LOGGER.error(f"Error storing cache entry: {e}")

    async def astore_cache_entry(
        self,
        query: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> None:
        """
        Asynchronously store a query-response pair in the semantic cache with optional metadata and filters.

        Args:
            query: The query string to cache.
            response: The response string to cache.
            metadata: Optional dictionary of metadata to associate with this cache entry.
            filters: Optional dictionary of filters to apply when retrieving this entry.
            ttl: Optional time-to-live for this cache entry in seconds (overrides default TTL if provided).

        Returns:
            None

        """
        try:
            key = await self.cache.astore(
                prompt=query,
                response=response,
                metadata=metadata,
                filters=filters,
                ttl=ttl if ttl else self.ttl,
            )
            LOGGER.info(f"Successfully stored cache entry with key: {key}")
        except Exception as e:
            LOGGER.error(f"Error storing cache entry: {e}")

    def remove_cache_entries(
        self, ids: Optional[List[str]] = None, keys: Optional[List[str]] = None
    ) -> None:
        """
        Remove specific entries from the cache by ID or Redis key.

        Args:
            ids (Optional[List[str]]): List of entry IDs to remove from the cache.
                Entry IDs are the unique identifiers without the cache prefix.
            keys (Optional[List[str]]): List of full Redis keys to remove from the cache.
                Keys are the complete Redis keys including the cache prefix.

        Note:
            At least one of ids or keys must be provided.

        Raises:
            ValueError: If neither ids nor keys is provided.

        """
        if ids is None and keys is None:
            raise ValueError("At least one of ids or keys must be provided.")
        try:
            self.cache.drop(ids=ids, keys=keys)

            if keys:
                for key in keys:
                    LOGGER.info(f"Successfully removed cache entry with key: {key}")
            else:
                for id in ids:
                    LOGGER.info(f"Successfully removed cache entry with ID: {id}")
        except Exception as e:
            LOGGER.error(f"Error removing cache entries: {e}")

    async def aremove_cache_entries(
        self, ids: Optional[List[str]] = None, keys: Optional[List[str]] = None
    ) -> None:
        """
        Asynchronously remove specific entries from the cache by ID or Redis key.

        Args:
            ids (Optional[List[str]]): List of entry IDs to remove from the cache.
                Entry IDs are the unique identifiers without the cache prefix.
            keys (Optional[List[str]]): List of full Redis keys to remove from the cache.
                Keys are the complete Redis keys including the cache prefix.

        Note:
            At least one of ids or keys must be provided.

        Raises:
            ValueError: If neither ids nor keys is provided.

        """
        if ids is None and keys is None:
            raise ValueError("At least one of ids or keys must be provided.")

        try:
            await self.cache.adrop(ids=ids, keys=keys)

            if keys:
                for key in keys:
                    LOGGER.info(f"Successfully removed cache entry with key: {key}")
            else:
                for id in ids:
                    LOGGER.info(f"Successfully removed cache entry with ID: {id}")
        except Exception as e:
            LOGGER.error(f"Error removing cache entries: {e}")

    def update(self, key: str, **kwargs) -> None:
        """
        Update specific fields within an existing cache entry. If no fields
        are passed, then only the document TTL is refreshed.

        Args:
            key (str): the key of the document to update using kwargs.

        Raises:
            ValueError if an incorrect mapping is provided as a kwarg.
            TypeError if metadata is provided and not of type dict.

        .. code-block:: python

            key = cache.store('this is a prompt', 'this is a response')
            cache.update(key, metadata={"hit_count": 1, "model_name": "Llama-2-7b"})

        """
        try:
            self.cache.update(key=key, **kwargs)
            LOGGER.info(f"Successfully updated cache entry with key: {key}")
        except Exception as e:
            LOGGER.error(f"Error updating cache entry with key {key}: {e}")

    async def aupdate(self, key: str, **kwargs) -> None:
        """
        Asynchronously update specific fields within an existing cache entry. If no fields
        are passed, then only the document TTL is refreshed.

        Args:
            key (str): the key of the document to update using kwargs.

        Raises:
            ValueError if an incorrect mapping is provided as a kwarg.
            TypeError if metadata is provided and not of type dict.

        .. code-block:: python

            key = await cache.astore('this is a prompt', 'this is a response')
            await cache.aupdate(
                key,
                metadata={"hit_count": 1, "model_name": "Llama-2-7b"}
            )

        """
        try:
            await self.cache.aupdate(key=key, **kwargs)
            LOGGER.info(f"Successfully updated cache entry with key: {key}")
        except Exception as e:
            LOGGER.error(f"Error updating cache entry with key {key}: {e}")

    @classmethod
    def from_config(
        cls,
        config,
    ) -> "RedisSemanticCache":
        """
        Construct a RedisSemanticCache from a config object with optional overrides.

        The config object should have attributes like:
        - redis_url (default: "redis://localhost:6379")
        - cache_name (default: "semantic-cache")
        - cache_distance_threshold (default: 0.3)
        - cache_ttl_seconds (default: 3600)

        Any of these can be overridden via kwargs:
        - redis_url, name, distance_threshold, ttl

        Example:
            # Use all config values
            cache = RedisSemanticCache.from_config(config)

            # Override specific values
            cache = RedisSemanticCache.from_config(config, name="custom-cache")

        """
        return cls(
            redis_url=config["redis_url"],
            name=config["cache_name"],
            distance_threshold=float(config["distance_threshold"]),
            ttl=int(config["ttl_seconds"]),
        )

    def get_all_cache_keys(self) -> List[str]:
        """Retrieve a list of all cache keys currently stored in Redis cache."""
        keys = []
        for key in self.redis_client.scan_iter():
            if key.startswith(self.name.encode("utf-8")):
                keys.append(key.decode("utf-8"))
        return keys

    def retrieve_cache_entry(self, key: str) -> dict:
        """Retrieve a cache entry by its key."""
        try:
            details = self.redis_client.hgetall(key)
            readable_dict = {}
            for key, value in details.items():
                k = key.decode("utf-8")
                if k == "prompt_vector":
                    floats = list(struct.unpack(f"{len(value) // 4}f", value))
                    readable_dict[k] = floats
                elif k == "metadata":
                    readable_dict[k] = json.loads(value.decode("utf-8"))
                else:
                    readable_dict[k] = value.decode("utf-8")

            return readable_dict
        except Exception as e:
            LOGGER.error(f"Error retrieving cache entry with key {key}: {e}")
            return None

    def check(
        self,
        query: str,
        distance_threshold: Optional[float] = None,
        num_results: int = 1,
    ) -> CacheResults:
        """
        Check semantic cache for a single query.

        Args:
            query: The query string to search for
            distance_threshold: Maximum semantic distance (lower = more similar)
            num_results: Maximum number of results to return

        Returns:
            List of CacheResult objects (empty list if no matches)

        """
        candidates = self.cache.check(
            query, distance_threshold=distance_threshold, num_results=num_results
        )

        if not candidates:
            return CacheResults(matches=[])

        results: List[CacheResult] = []
        for item in candidates[:num_results]:
            result = dict(item)
            result["id"] = result.get("entry_id", "")
            result["query"] = result.get("prompt", "")
            result["vector_distance"] = float(result.get("vector_distance", 0.0))
            result["cosine_similarity"] = float((2 - result["vector_distance"]) / 2)

            results.append(CacheResult(**result))

        LOGGER.info(
            f"Found {len(results)} cache matches for query: '{query}' with distance threshold: {distance_threshold}"
        )

        return CacheResults(matches=results)

    async def acheck(
        self,
        query: str,
        distance_threshold: Optional[float] = None,
        num_results: int = 1,
    ) -> List[CacheResult]:
        """
        Asynchronously check semantic cache for a single query.

        Args:
            query: The query string to search for
            distance_threshold: Maximum semantic distance (lower = more similar)
            num_results: Maximum number of results to return

        Returns:
            List of CacheResult objects (empty list if no matches)

        """
        candidates = await self.cache.acheck(
            query,
            distance_threshold=distance_threshold,
            num_results=num_results,
        )

        if not candidates:
            return CacheResults(query=query, matches=[])

        results: List[CacheResult] = []
        for item in candidates[:num_results]:
            result = dict(item)
            result["id"] = result.get("entry_id", "")
            result["vector_distance"] = float(result.get("vector_distance", 0.0))
            result["cosine_similarity"] = float((2 - result["vector_distance"]) / 2)
            result["query"] = query

            results.append(CacheResult(**result))

        LOGGER.info(
            f"Found {len(results)} cache matches for query: '{query}' with distance threshold: {distance_threshold}"
        )

        return CacheResults(matches=results)

    def clear_cache(self) -> None:
        """Clear all entries from the semantic cache."""
        self.cache.clear()
        LOGGER.info("Cleared all entries from the semantic cache.")

    async def aclear_cache(self) -> None:
        """Asynchronously clear all entries from the semantic cache."""
        await self.cache.aclear()
        LOGGER.info("Cleared all entries from the semantic cache.")

    def delete_cache(self) -> None:
        """Delete the cache and its index entirely."""
        self.cache.delete()
        LOGGER.info("Deleted the cache and its index entirely.")

    async def adelete_cache(self) -> None:
        """Asynchronously delete the cache and its index entirely."""
        await self.cache.adelete()
        LOGGER.info("Deleted the cache and its index entirely.")
