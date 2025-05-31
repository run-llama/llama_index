import logging
from typing import Dict, List, Optional, Tuple, Type
from cachetools import Cache

from llama_index.core.storage.kvstore.types import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTION,
    BaseKVStore,
)

logger = logging.getLogger(__name__)


class CachetoolsKVStore(BaseKVStore):
    """
    Cachetools Key-Value store.

    Args:
        cache (Cache): the cachetools cache to use as key-value store.

    """

    def __init__(self, cache_cls: Type[Cache], *args, **kwargs) -> None:
        self.collections_caches = {}
        self.cache_cls = cache_cls
        self.cache_args = args
        self.cache_kwargs = kwargs

    def _get_collection_cache(self, collection: str) -> Cache:
        if collection not in self.collections_caches:
            self.collections_caches[collection] = self.cache_cls(
                *self.cache_args, **self.cache_kwargs
            )
        return self.collections_caches[collection]

    def put(
        self,
        key: str,
        val: dict,
        collection: str = DEFAULT_COLLECTION,
    ) -> None:
        """
        Put a key-value pair into the store.

        Args:
            key (str): key
            val (dict): value
            collection (str): collection name

        """
        self.put_all([(key, val)], collection=collection)

    async def aput(
        self,
        key: str,
        val: dict,
        collection: str = DEFAULT_COLLECTION,
    ) -> None:
        """
        Put a key-value pair into the store.

        Args:
            key (str): key
            val (dict): value
            collection (str): collection name

        """
        await self.aput_all([(key, val)], collection=collection)

    def put_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        cache = self._get_collection_cache(collection)

        if len(kv_pairs) > cache.maxsize:
            msg = "The number of key-value pairs (%d) is greater than the cache size (%d).".format(
                len(kv_pairs),
                cache.maxsize,
            )
            raise ValueError(msg)

        for key, val in kv_pairs:
            cache[key] = val

    async def aput_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.put_all(kv_pairs=kv_pairs, collection=collection, batch_size=batch_size)

    def get(self, key: str, collection: str = DEFAULT_COLLECTION) -> Optional[dict]:
        """
        Get a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        cache = self._get_collection_cache(collection)

        return cache[key] if key in cache else None

    async def aget(
        self, key: str, collection: str = DEFAULT_COLLECTION
    ) -> Optional[dict]:
        """
        Get a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        return self.get(key=key, collection=collection)

    def get_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        """
        Get all values from the store.

        Args:
            collection (str): collection name

        """
        cache = self._get_collection_cache(collection)

        return dict(cache.items())

    async def aget_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        """
        Get all values from the store.

        Args:
            collection (str): collection name

        """
        return self.get_all(collection=collection)

    def delete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """
        Delete a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        cache = self._get_collection_cache(collection)

        return bool(cache.pop(key, None))

    async def adelete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """
        Delete a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        return self.delete(key=key, collection=collection)
