import pytest

from llama_index.core.ingestion.interceptors import AsyncStreamRetryInterceptor


class FlakyError(Exception):
    status = 503


def flaky_producer_factory(failures: int):
    calls = {"n": 0}

    async def _gen():
        calls["n"] += 1
        if calls["n"] <= failures:
            raise FlakyError("temporary outage")
        yield 1
        yield 2

    return _gen


@pytest.mark.asyncio
async def test_recovers_after_transient_failures():
    interceptor = AsyncStreamRetryInterceptor(max_retries=3, backoff_base=0.01, max_backoff=0.05, jitter=0.0)
    results = []
    async for item in interceptor.run(flaky_producer_factory(failures=2)):
        results.append(item)
    assert results == [1, 2]


@pytest.mark.asyncio
async def test_gives_up_when_exceeding_retries():
    interceptor = AsyncStreamRetryInterceptor(max_retries=1, backoff_base=0.01, jitter=0.0)
    with pytest.raises(FlakyError):
        async for _ in interceptor.run(flaky_producer_factory(failures=3)):
            pass
