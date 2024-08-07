import json
from typing import Any, Dict, List, Optional, Tuple, cast

from llama_index.legacy.storage.kvstore.types import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTION,
    BaseKVStore,
)

IMPORT_ERROR_MSG = "`redis` package not found, please run `pip install redis`"


class RedisKVStore(BaseKVStore):
    """Redis KV Store.

    Args:
        redis_client (Any): Redis client
        redis_aclient (Any): Redis Async client
        redis_url (Optional[str]): Redis server URI
        async_client (Optional[bool]): Whether to use async client or not when passing redis_url (default: False)

    Raises:
            ValueError: If redis-py is not installed

    Examples:
        >>> from llama_index.legacy.storage.kvstore.redis_kvstore import RedisKVStore
        >>> # Create a RedisKVStore
        >>> redis_kv_store = RedisKVStore(
        >>>     redis_url="redis://127.0.0.1:6379")

    """

    def __init__(
        self,
        redis_uri: Optional[str] = "redis://127.0.0.1:6379",
        **kwargs: Any,
    ) -> None:
        try:
            from redis import Redis
            from redis.asyncio import RedisAsync
        except ImportError:
            raise ValueError(IMPORT_ERROR_MSG)

        # user could inject customized redis client.
        # for instance, redis have specific TLS connection, etc.
        if "redis_client" in kwargs:
            self._redis_client = cast(Redis, kwargs["redis_client"])
        elif "redis_aclient" in kwargs:
            self._redis_aclient = cast(RedisAsync, kwargs["redis_aclient"])
        elif redis_uri is not None:
            # otherwise, try initializing redis client
            try:
                # connect to redis from url
                if kwargs["async_client"] is True:
                    self._redis_aclient = RedisAsync.from_url(redis_uri, **kwargs)
                else:
                    self._redis_client = Redis.from_url(redis_uri, **kwargs)
            except ValueError as e:
                raise ValueError(f"Redis failed to connect: {e}")
        else:
            raise ValueError("Either 'redis_client' or redis_url must be provided.")

    def _check_async_client(self) -> None:
        if self._redis_aclient is None:
            raise ValueError("RedisKVStore was not initialized with an async client")

    def put(self, key: str, val: dict, collection: str = DEFAULT_COLLECTION) -> None:
        """Put a key-value pair into the store.

        Args:
            key (str): key
            val (dict): value
            collection (str): collection name

        """
        self._redis_client.hset(name=collection, key=key, value=json.dumps(val))

    async def aput(
        self, key: str, val: dict, collection: str = DEFAULT_COLLECTION
    ) -> None:
        """Put a key-value pair into the store.

        Args:
            key (str): key
            val (dict): value
            collection (str): collection name

        """
        self._check_async_client()
        await self._redis_aclient.hset(name=collection, key=key, value=json.dumps(val))

    def put_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Put a dictionary of key-value pairs into the store.

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
        """Get a value from the store.

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
        """Get a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        val_str = await self._redis_aclient.hget(name=collection, key=key)
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
        self._check_async_client()
        collection_kv_dict = {}
        async for key, val_str in self._redis_aclient.hscan_iter(name=collection):
            value = dict(json.loads(val_str))
            collection_kv_dict[key.decode()] = value
        return collection_kv_dict

    def delete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """Delete a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        deleted_num = self._redis_client.hdel(collection, key)
        return bool(deleted_num > 0)

    async def adelete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """Delete a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        self._check_async_client()
        deleted_num = await self._redis_aclient.hdel(collection, key)
        return bool(deleted_num > 0)

    @classmethod
    def from_host_and_port(
        cls,
        host: str,
        port: int,
        async_client: bool = False,
    ) -> "RedisKVStore":
        """Load a RedisKVStore from a Redis host and port.

        Args:
            host (str): Redis host
            port (int): Redis port
        """
        url = f"redis://{host}:{port}".format(host=host, port=port)
        return cls(redis_uri=url, async_client=async_client)

    @classmethod
    def from_redis_client(cls, redis_client: Any) -> "RedisKVStore":
        """Load a RedisKVStore from a Redis Client.

        Args:
            redis_client (Redis): Redis client
        """
        return cls(redis_client=redis_client)

    @classmethod
    def from_redis_aclient(cls, redis_aclient: Any) -> "RedisKVStore":
        """Load a RedisKVStore from a Redis Async Client.

        Args:
            redis_client (Redis): Redis client
        """
        return cls(redis_aclient=redis_aclient)
