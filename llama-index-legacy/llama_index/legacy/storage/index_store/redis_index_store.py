from typing import Any, Optional

from llama_index.legacy.storage.index_store.keyval_index_store import KVIndexStore
from llama_index.legacy.storage.kvstore.redis_kvstore import RedisKVStore


class RedisIndexStore(KVIndexStore):
    """Redis Index store.

    Args:
        redis_kvstore (RedisKVStore): Redis key-value store
        namespace (str): namespace for the index store

    """

    def __init__(
        self,
        redis_kvstore: RedisKVStore,
        namespace: Optional[str] = None,
    ) -> None:
        """Init a RedisIndexStore."""
        super().__init__(redis_kvstore, namespace=namespace)
        # avoid conflicts with redis docstore
        self._collection = f"{self._namespace}/index"

    @classmethod
    def from_redis_client(
        cls,
        redis_client: Any,
        namespace: Optional[str] = None,
    ) -> "RedisIndexStore":
        """Load a RedisIndexStore from a Redis Client."""
        redis_kvstore = RedisKVStore.from_redis_client(redis_client=redis_client)
        return cls(redis_kvstore, namespace)

    @classmethod
    def from_host_and_port(
        cls,
        host: str,
        port: int,
        namespace: Optional[str] = None,
    ) -> "RedisIndexStore":
        """Load a RedisIndexStore from a Redis host and port."""
        redis_kvstore = RedisKVStore.from_host_and_port(host, port)
        return cls(redis_kvstore, namespace)
