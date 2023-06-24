from typing import Optional
from llama_index.storage.index_store.keyval_index_store import KVIndexStore
from llama_index.storage.kvstore.redis_kvstore import RedisKVStore
from redis import Redis


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

    @classmethod
    def from_uri(
        cls,
        redis_client: Redis,
        namespace: Optional[str] = None,
    ) -> "RedisIndexStore":
        """Load a RedisIndexStore from a Redis URI."""
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
