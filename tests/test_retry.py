import time
import pytest
from llama_index.retry import ExponentialBackoffRetryStrategy


def test_no_retries():
    retry_strategy = ExponentialBackoffRetryStrategy()
    counter = 0

    @retry_strategy.decorate
    def increment_counter():
        nonlocal counter
        counter += 1

    increment_counter()

    assert counter == 1


def test_retry_on_failure():
    """
    Test that the retry mechanism is working by ensuring that a function
    that throws an exception is called the expected number of times before
    it finally fails.
    """
    retry_strategy = ExponentialBackoffRetryStrategy(
        num_attempts=3, min_wait=0.01, max_wait=0.02
    )
    counter = 0

    @retry_strategy.decorate
    def increment_counter():
        nonlocal counter
        counter += 1
        raise Exception("Test exception")

    with pytest.raises(Exception):
        increment_counter()

    assert counter == 3


def test_wait_time():
    """
    Test that the wait time between each retry is increasing as expected.
    """
    retry_strategy = ExponentialBackoffRetryStrategy(
        num_attempts=3, min_wait=0.01, max_wait=0.02
    )
    counter = 0
    times = []

    @retry_strategy.decorate
    def track_time():
        nonlocal counter, times
        times.append(time.time())
        counter += 1
        raise Exception("Test exception")

    with pytest.raises(Exception):
        track_time()

    assert all(
        t2 - t1 >= 0.01 * (2**i)
        for i, (t1, t2) in enumerate(zip(times[:-1], times[1:]))
    ), "Exponential backoff not respected."


def test_max_retries():
    """
    Test that the maximum number of retries is correctly enforced.
    """
    retry_strategy = ExponentialBackoffRetryStrategy(
        num_attempts=3, min_wait=0.01, max_wait=0.02
    )
    counter = 0

    @retry_strategy.decorate
    def increment_counter():
        nonlocal counter
        counter += 1
        raise Exception("Test exception")

    with pytest.raises(Exception):
        increment_counter()

    assert counter == 3


@pytest.mark.asyncio
async def test_async_retry_on_failure():
    """
    Test that the retry mechanism works as expected when decorating asynchronous functions.
    """
    retry_strategy = ExponentialBackoffRetryStrategy(
        num_attempts=3, min_wait=0.01, max_wait=0.02
    )
    counter = 0

    @retry_strategy.decorate
    async def increment_counter():
        nonlocal counter
        counter += 1
        raise Exception("Test exception")

    with pytest.raises(Exception):
        await increment_counter()

    assert counter == 3


def test_retry_on_specific_exception():
    """
    Test that the retry mechanism only retries on specific exceptions.
    """
    retry_strategy = ExponentialBackoffRetryStrategy(
        num_attempts=3, min_wait=0.01, max_wait=0.02, retry_on_exceptions=[ValueError]
    )
    counter = 0

    @retry_strategy.decorate
    def raise_exception():
        nonlocal counter
        counter += 1
        if counter == 1:
            raise ValueError("Test exception")
        elif counter == 2:
            raise NotImplementedError("Test exception")

    with pytest.raises(NotImplementedError):
        raise_exception()

    assert (
        counter == 2
    ), "Retry mechanism retried on an exception not in the retry_on_exceptions list."


@pytest.mark.asyncio
async def test_async_retry_on_specific_exception():
    """
    Test that the retry mechanism only retries on specific exceptions for asynchronous functions.
    """
    retry_strategy = ExponentialBackoffRetryStrategy(
        num_attempts=3, min_wait=0.01, max_wait=0.02, retry_on_exceptions=[ValueError]
    )
    counter = 0

    @retry_strategy.decorate
    async def raise_exception():
        nonlocal counter
        counter += 1
        if counter == 1:
            raise ValueError("Test exception")
        elif counter == 2:
            raise NotImplementedError("Test exception")

    with pytest.raises(NotImplementedError):
        await raise_exception()

    assert (
        counter == 2
    ), "Retry mechanism retried on an exception not in the retry_on_exceptions list."
