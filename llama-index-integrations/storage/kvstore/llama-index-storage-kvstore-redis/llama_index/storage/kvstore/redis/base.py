import json
from typing import Any, Dict, List, Optional, Tuple

from llama_index.core.storage.kvstore.types import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTION,
    BaseKVStore,
)

from redis.asyncio import Redis as AsyncRedis
from redis import Redis


class RedisKVStore(BaseKVStore):
    """
    Redis KV Store.

    Args:
        redis_uri (str): Redis URI
        redis_client (Any): Redis client
        async_redis_client (Any): Async Redis client

    Raises:
            ValueError: If redis-py is not installed

    Examples:
        >>> from llama_index.storage.kvstore.redis import RedisKVStore
        >>> # Create a RedisKVStore
        >>> redis_kv_store = RedisKVStore(
        >>>     redis_url="redis://127.0.0.1:6379")

    """

    def __init__(
        self,
        redis_uri: Optional[str] = "redis://127.0.0.1:6379",
        redis_client: Optional[Redis] = None,
        async_redis_client: Optional[AsyncRedis] = None,
        **kwargs: Any,
    ) -> None:
        # user could inject customized redis client.
        # for instance, redis have specific TLS connection, etc.
        if redis_client is not None:
            self._redis_client = redis_client

            # create async client from sync client
            if async_redis_client is not None:
                self._async_redis_client = async_redis_client
            else:
                try:
                    self._async_redis_client = AsyncRedis.from_url(
                        self._redis_client.connection_pool.connection_kwargs["url"]
                    )
                except Exception:
                    print(
                        "Could not create async redis client from sync client, "
                        "pass in `async_redis_client` explicitly."
                    )
                    self._async_redis_client = None
        elif redis_uri is not None:
            # otherwise, try initializing redis client
            try:
                # connect to redis from url
                self._redis_client = Redis.from_url(redis_uri, **kwargs)
                self._async_redis_client = AsyncRedis.from_url(redis_uri, **kwargs)
            except ValueError as e:
                raise ValueError(f"Redis failed to connect: {e}")
        else:
            raise ValueError("Either 'redis_client' or redis_url must be provided.")

    def put(self, key: str, val: dict, collection: str = DEFAULT_COLLECTION) -> None:
        """
        Put a key-value pair into the store.

        Args:
            key (str): key
            val (dict): value
            collection (str): collection name

        """
        self._redis_client.hset(name=collection, key=key, value=json.dumps(val))

    async def aput(
        self, key: str, val: dict, collection: str = DEFAULT_COLLECTION
    ) -> None:
        """
        Put a key-value pair into the store.

        Args:
            key (str): key
            val (dict): value
            collection (str): collection name

        """
        await self._async_redis_client.hset(
            name=collection, key=key, value=json.dumps(val)
        )

    def put_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """
        Put a dictionary of key-value pairs into the store.

        Args:
            kv_pairs (List[Tuple[str, dict]]): key-value pairs
            collection (str): collection name

        """
        with self._redis_client.pipeline() as pipe:
            cur_batch = 0
            for key, val in kv_pairs:
                pipe.hset(name=collection, key=key, value=json.dumps(val))
                cur_batch += 1

                if cur_batch >= batch_size:
                    cur_batch = 0
                    pipe.execute()

            if cur_batch > 0:
                pipe.execute()

    def get(self, key: str, collection: str = DEFAULT_COLLECTION) -> Optional[dict]:
        """
        Get a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        val_str = self._redis_client.hget(name=collection, key=key)
        if val_str is None:
            return None
        return json.loads(val_str)

    async def aget(
        self, key: str, collection: str = DEFAULT_COLLECTION
    ) -> Optional[dict]:
        """
        Get a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        val_str = await self._async_redis_client.hget(name=collection, key=key)
        if val_str is None:
            return None
        return json.loads(val_str)

    def get_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        """Get all values from the store."""
        collection_kv_dict = {}
        for key, val_str in self._redis_client.hscan_iter(name=collection):
            value = dict(json.loads(val_str))
            collection_kv_dict[key.decode()] = value
        return collection_kv_dict

    async def aget_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        """Get all values from the store."""
        collection_kv_dict = {}
        async for key, val_str in self._async_redis_client.hscan_iter(name=collection):
            value = dict(json.loads(val_str))
            collection_kv_dict[key.decode()] = value
        return collection_kv_dict

    def delete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """
        Delete a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        deleted_num = self._redis_client.hdel(collection, key)
        return bool(deleted_num > 0)

    async def adelete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """
        Delete a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        deleted_num = await self._async_redis_client.hdel(collection, key)
        return bool(deleted_num > 0)

    @classmethod
    def from_host_and_port(
        cls,
        host: str,
        port: int,
    ) -> "RedisKVStore":
        """
        Load a RedisKVStore from a Redis host and port.

        Args:
            host (str): Redis host
            port (int): Redis port

        """
        url = f"redis://{host}:{port}".format(host=host, port=port)
        return cls(redis_uri=url)

    @classmethod
    def from_redis_client(cls, redis_client: Any) -> "RedisKVStore":
        """
        Load a RedisKVStore from a Redis Client.

        Args:
            redis_client (Redis): Redis client

        """
        return cls(redis_client=redis_client)
