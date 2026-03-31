import time
import pytest
from redis import Redis, ConnectionError
from redis.asyncio import Redis as RedisAsync
import docker


@pytest.fixture(scope="session", autouse=True)
def redis_server():
    client = docker.from_env()

    try:
        old_container = client.containers.get("redis_test_env")
        old_container.stop()
        old_container.remove()
    except docker.errors.NotFound:
        pass

    container = client.containers.run(
        "redis/redis-stack:latest",
        detach=True,
        name="redis_test_env",
        ports={"6379/tcp": 6379},
    )

    r = Redis(host="localhost", port=6379)
    retries = 5
    while retries > 0:
        try:
            if r.ping():
                break
        except ConnectionError:
            retries -= 1
            time.sleep(3)

    yield container  # This is where your tests run

    container.stop()
    container.remove()


@pytest.fixture()
def redis_client() -> Redis:
    return Redis.from_url("redis://localhost:6379/0")


@pytest.fixture()
def redis_client_async() -> RedisAsync:
    """Fixture that provides an asynchronous Redis client."""
    return RedisAsync.from_url("redis://localhost:6379/0")
