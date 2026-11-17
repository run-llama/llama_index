"""
The async client derived from an injected sync client must reach the same server.

None of these tests connect: redis-py clients and connection objects are lazy,
so the settings can be read off them without a server.
"""

from typing import Any

import pytest
from redis import BlockingConnectionPool, ConnectionPool, Redis
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.connection import (
    SSLConnection as AsyncSSLConnection,
    UnixDomainSocketConnection as AsyncUnixDomainSocketConnection,
)
from redis.asyncio.retry import Retry as AsyncRetry
from redis.backoff import ExponentialBackoff
from redis.connection import Connection
from redis.retry import Retry

from llama_index.storage.kvstore.redis import RedisKVStore
from llama_index.storage.kvstore.redis.base import _async_kwargs_from_sync_client


def _async_connection(redis_client: Redis) -> Any:
    store = RedisKVStore(redis_client=redis_client)
    async_client = store._async_redis_client
    assert isinstance(async_client, AsyncRedis)
    pool = async_client.connection_pool
    return pool.connection_class(**pool.connection_kwargs)


def test_plain_url() -> None:
    conn = _async_connection(Redis.from_url("redis://user:secret@cache.local:6380/2"))

    assert not isinstance(conn, (AsyncSSLConnection, AsyncUnixDomainSocketConnection))
    assert conn.host == "cache.local"
    assert conn.port == 6380
    assert conn.db == 2
    assert conn.username == "user"
    assert conn.password == "secret"


def test_rediss_url_keeps_tls() -> None:
    conn = _async_connection(Redis.from_url("rediss://:secret@cache.local:6390/1"))

    assert isinstance(conn, AsyncSSLConnection)
    assert conn.host == "cache.local"
    assert conn.port == 6390
    assert conn.db == 1
    assert conn.password == "secret"


def test_ssl_kwargs_survive() -> None:
    conn = _async_connection(
        Redis(
            host="cache.local",
            port=6390,
            ssl=True,
            ssl_ca_certs="/etc/ssl/ca.pem",
            ssl_cert_reqs="required",
        )
    )

    assert isinstance(conn, AsyncSSLConnection)
    assert conn.ssl_context.ca_certs == "/etc/ssl/ca.pem"


def test_unix_socket() -> None:
    conn = _async_connection(Redis.from_url("unix:///tmp/redis.sock?db=3"))

    assert isinstance(conn, AsyncUnixDomainSocketConnection)
    assert conn.path == "/tmp/redis.sock"
    assert conn.db == 3


def test_resp3_protocol() -> None:
    conn = _async_connection(Redis(host="cache.local", protocol=3))

    assert int(conn.protocol) == 3


def test_custom_retry_becomes_async_retry() -> None:
    conn = _async_connection(
        Redis(host="cache.local", retry=Retry(ExponentialBackoff(), 5))
    )

    assert isinstance(conn.retry, AsyncRetry)
    assert conn.retry._retries == 5
    assert isinstance(conn.retry._backoff, ExponentialBackoff)


def test_max_connections_survive() -> None:
    store = RedisKVStore(redis_client=Redis(host="cache.local", max_connections=7))

    assert store._async_redis_client.connection_pool.max_connections == 7


def test_explicit_async_client_is_kept() -> None:
    async_client = AsyncRedis(host="other.local")
    store = RedisKVStore(
        redis_client=Redis(host="cache.local"), async_redis_client=async_client
    )

    assert store._async_redis_client is async_client


class _CustomConnection(Connection):
    pass


def _connect_callback(*args: Any) -> None:
    pass


@pytest.mark.parametrize(
    ("redis_client", "reason"),
    [
        pytest.param(
            Redis(
                connection_pool=ConnectionPool(
                    connection_class=_CustomConnection, host="cache.local"
                )
            ),
            "unsupported connection class",
            id="custom-connection-class",
        ),
        pytest.param(
            Redis(
                connection_pool=BlockingConnectionPool(
                    host="cache.local", max_connections=5
                )
            ),
            "unsupported connection pool",
            id="blocking-pool",
        ),
        pytest.param(
            Redis(host="cache.local", redis_connect_func=_connect_callback),
            "redis_connect_func",
            id="sync-connect-callback",
        ),
        pytest.param(
            Redis(host="cache.local", ssl=True, ssl_validate_ocsp=True),
            "ssl_validate_ocsp",
            id="tls-setting-async-lacks",
        ),
    ],
)
def test_unmatchable_client_leaves_async_unset(
    redis_client: Redis, reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        _async_kwargs_from_sync_client(redis_client)

    store = RedisKVStore(redis_client=redis_client)

    assert store._async_redis_client is None
