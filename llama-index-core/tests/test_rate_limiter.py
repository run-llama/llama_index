"""Tests for the rate limiter module."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms.mock import MockLLM
from llama_index.core.rate_limiter import (
    BaseRateLimiter,
    RateLimiter,
    SlidingWindowRateLimiter,
    TokenBucketRateLimiter,
)


# ---------------------------------------------------------------------------
# BaseRateLimiter contract tests
# ---------------------------------------------------------------------------


def test_base_rate_limiter_is_abstract() -> None:
    with pytest.raises(TypeError, match="abstract method"):
        BaseRateLimiter()


def test_token_bucket_is_subclass_of_base() -> None:
    assert issubclass(TokenBucketRateLimiter, BaseRateLimiter)


def test_rate_limiter_alias_is_token_bucket() -> None:
    assert RateLimiter is TokenBucketRateLimiter


def test_instance_is_base_rate_limiter() -> None:
    rl = TokenBucketRateLimiter(requests_per_minute=60)
    assert isinstance(rl, BaseRateLimiter)


def test_custom_rate_limiter_subclass() -> None:
    """Users can create custom rate limiters by subclassing BaseRateLimiter."""

    class FixedDelayLimiter(BaseRateLimiter):
        def acquire(self, num_tokens: int = 0) -> None:
            pass

        async def async_acquire(self, num_tokens: int = 0) -> None:
            pass

    limiter = FixedDelayLimiter()
    assert isinstance(limiter, BaseRateLimiter)
    limiter.acquire()


# ---------------------------------------------------------------------------
# Token-bucket algorithm tests
# ---------------------------------------------------------------------------


def test_creation_rpm_only() -> None:
    rl = RateLimiter(requests_per_minute=60)
    assert rl.requests_per_minute == 60
    assert rl.tokens_per_minute is None


def test_creation_tpm_only() -> None:
    rl = RateLimiter(tokens_per_minute=10000)
    assert rl.tokens_per_minute == 10000
    assert rl.requests_per_minute is None


def test_creation_both() -> None:
    rl = RateLimiter(requests_per_minute=30, tokens_per_minute=5000)
    assert rl.requests_per_minute == 30
    assert rl.tokens_per_minute == 5000


def test_validation_rejects_zero() -> None:
    with pytest.raises(Exception):
        RateLimiter(requests_per_minute=0)


def test_validation_rejects_negative() -> None:
    with pytest.raises(Exception):
        RateLimiter(tokens_per_minute=-1)


def test_burst_within_limit() -> None:
    """Requests within the bucket size should not block."""
    rl = RateLimiter(requests_per_minute=10)
    start = time.monotonic()
    for _ in range(10):
        rl.acquire()
    elapsed = time.monotonic() - start
    assert elapsed < 1.0


def test_acquire_blocks_when_exhausted() -> None:
    """After draining the bucket, acquire blocks until a token refills."""
    rl = RateLimiter(requests_per_minute=60)
    # Drain the bucket
    for _ in range(60):
        rl.acquire()

    with (
        patch("llama_index.core.rate_limiter.time.sleep") as mock_sleep,
        patch("llama_index.core.rate_limiter.time.monotonic") as mock_time,
    ):
        base = 1000.0
        # Align internal clock with mock timeline
        rl._last_refill_time = base
        rl._request_tokens = 0.0
        # First monotonic(): no elapsed time, bucket empty -> must sleep
        # Second monotonic(): 2s elapsed -> refills 2 tokens -> acquire succeeds
        mock_time.side_effect = [base, base + 2.0]
        mock_sleep.return_value = None
        rl.acquire()
        mock_sleep.assert_called_once()


def test_refill_caps_at_max() -> None:
    """Token count must not exceed the configured maximum."""
    rl = RateLimiter(requests_per_minute=10)
    # Simulate long idle period
    rl._last_refill_time = time.monotonic() - 3600
    rl._request_tokens = 0.0
    rl.acquire()
    # After refill (capped at 10) minus 1 consumed
    assert rl._request_tokens <= 9.0
    assert rl._request_tokens >= 8.5


@pytest.mark.asyncio
async def test_async_acquire_burst_within_limit() -> None:
    rl = RateLimiter(requests_per_minute=10)
    start = time.monotonic()
    for _ in range(10):
        await rl.async_acquire()
    elapsed = time.monotonic() - start
    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_async_acquire_tpm_limiting() -> None:
    """TPM limiting should throttle based on token count."""
    rl = RateLimiter(tokens_per_minute=100)
    await rl.async_acquire(num_tokens=100)

    with (
        patch(
            "llama_index.core.rate_limiter.asyncio.sleep",
            new_callable=AsyncMock,
        ) as mock_sleep,
        patch("llama_index.core.rate_limiter.time.monotonic") as mock_time,
    ):
        base = 1000.0
        # Align internal clock with mock timeline
        rl._last_refill_time = base
        rl._token_tokens = 0.0
        # First monotonic(): no elapsed time, need 50 tokens but 0 available -> sleep
        # Second monotonic(): 60s elapsed -> refills 100 tokens -> acquire succeeds
        mock_time.side_effect = [base, base + 60.0]
        await rl.async_acquire(num_tokens=50)
        mock_sleep.assert_called_once()


@pytest.mark.asyncio
async def test_concurrent_async_rate_limiting() -> None:
    """Multiple concurrent async_acquire calls must all complete."""
    rl = RateLimiter(requests_per_minute=600)
    results: list = []

    async def worker(n: int) -> None:
        await rl.async_acquire()
        results.append(n)

    tasks = [worker(i) for i in range(20)]
    await asyncio.gather(*tasks)
    assert len(results) == 20


# ---------------------------------------------------------------------------
# LLM integration tests
# ---------------------------------------------------------------------------


def test_llm_sync_chat_calls_acquire() -> None:
    mock_limiter = MagicMock()
    llm = MockLLM()
    llm.rate_limiter = mock_limiter
    llm.chat([ChatMessage(role="user", content="hello")])
    # chat() triggers callback; internally calls complete() which also triggers
    assert mock_limiter.acquire.call_count >= 1


@pytest.mark.asyncio
async def test_llm_async_chat_calls_async_acquire() -> None:
    mock_limiter = MagicMock()
    mock_limiter.async_acquire = AsyncMock()
    llm = MockLLM()
    llm.rate_limiter = mock_limiter
    await llm.achat([ChatMessage(role="user", content="hello")])
    mock_limiter.async_acquire.assert_called_once()


def test_llm_sync_complete_calls_acquire() -> None:
    mock_limiter = MagicMock()
    llm = MockLLM()
    llm.rate_limiter = mock_limiter
    llm.complete("hello")
    mock_limiter.acquire.assert_called_once()


@pytest.mark.asyncio
async def test_llm_async_complete_calls_async_acquire() -> None:
    mock_limiter = MagicMock()
    mock_limiter.async_acquire = AsyncMock()
    llm = MockLLM()
    llm.rate_limiter = mock_limiter
    await llm.acomplete("hello")
    mock_limiter.async_acquire.assert_called_once()


def test_llm_without_rate_limiter_works() -> None:
    llm = MockLLM()
    assert llm.rate_limiter is None
    response = llm.complete("hello")
    assert response.text == "hello"


# ---------------------------------------------------------------------------
# Embedding integration tests
# ---------------------------------------------------------------------------


def test_embedding_single_calls_acquire() -> None:
    from llama_index.core.embeddings.mock_embed_model import MockEmbedding

    mock_limiter = MagicMock()
    embed = MockEmbedding(embed_dim=8, rate_limiter=mock_limiter)
    embed.get_text_embedding("test")
    mock_limiter.acquire.assert_called_once()


def test_embedding_batch_calls_acquire_per_batch() -> None:
    from llama_index.core.embeddings.mock_embed_model import MockEmbedding

    mock_limiter = MagicMock()
    embed = MockEmbedding(embed_dim=8, rate_limiter=mock_limiter, embed_batch_size=5)
    texts = ["text"] * 12  # 3 batches: 5, 5, 2
    embed.get_text_embedding_batch(texts)
    assert mock_limiter.acquire.call_count == 3


def test_embedding_without_rate_limiter_works() -> None:
    from llama_index.core.embeddings.mock_embed_model import MockEmbedding

    embed = MockEmbedding(embed_dim=8)
    assert embed.rate_limiter is None
    result = embed.get_text_embedding("test")
    assert len(result) == 8


def test_shared_rate_limiter_across_instances() -> None:
    from llama_index.core.embeddings.mock_embed_model import MockEmbedding

    rl = RateLimiter(requests_per_minute=100)
    embed1 = MockEmbedding(embed_dim=8, rate_limiter=rl)
    embed2 = MockEmbedding(embed_dim=8, rate_limiter=rl)
    assert embed1.rate_limiter is embed2.rate_limiter


def test_shared_limiter_between_llm_and_embedding() -> None:
    from llama_index.core.embeddings.mock_embed_model import MockEmbedding

    rl = RateLimiter(requests_per_minute=100)
    llm = MockLLM()
    llm.rate_limiter = rl
    embed = MockEmbedding(embed_dim=8, rate_limiter=rl)
    assert llm.rate_limiter is embed.rate_limiter


# ---------------------------------------------------------------------------
# SlidingWindowRateLimiter tests
# ---------------------------------------------------------------------------


def test_sliding_window_is_subclass_of_base() -> None:
    assert issubclass(SlidingWindowRateLimiter, BaseRateLimiter)


def test_sliding_window_instance_is_base_rate_limiter() -> None:
    rl = SlidingWindowRateLimiter(requests_per_minute=60)
    assert isinstance(rl, BaseRateLimiter)


def test_sliding_window_creation_rpm_only() -> None:
    rl = SlidingWindowRateLimiter(requests_per_minute=60)
    assert rl.requests_per_minute == 60
    assert rl.tokens_per_minute is None


def test_sliding_window_creation_tpm_only() -> None:
    rl = SlidingWindowRateLimiter(tokens_per_minute=10000)
    assert rl.tokens_per_minute == 10000
    assert rl.requests_per_minute is None


def test_sliding_window_creation_both() -> None:
    rl = SlidingWindowRateLimiter(
        requests_per_minute=30,
        tokens_per_minute=5000,
    )
    assert rl.requests_per_minute == 30
    assert rl.tokens_per_minute == 5000


def test_sliding_window_rejects_both_none() -> None:
    with pytest.raises(ValueError, match="At least one of"):
        SlidingWindowRateLimiter()


def test_sliding_window_rejects_zero_rpm() -> None:
    with pytest.raises(ValueError):
        SlidingWindowRateLimiter(requests_per_minute=0)


def test_sliding_window_rejects_negative_tpm() -> None:
    with pytest.raises(ValueError):
        SlidingWindowRateLimiter(tokens_per_minute=-1)


def test_sliding_window_burst_within_limit() -> None:
    """First N requests within the window should not block."""
    rl = SlidingWindowRateLimiter(requests_per_minute=10)
    start = time.monotonic()
    for _ in range(10):
        rl.acquire()
    elapsed = time.monotonic() - start
    assert elapsed < 1.0


def test_sliding_window_blocks_after_limit() -> None:
    """After exhausting the window, acquire blocks until oldest request exits."""
    rl = SlidingWindowRateLimiter(requests_per_minute=3)
    for _ in range(3):
        rl.acquire()

    with (
        patch("llama_index.core.rate_limiter.time.sleep") as mock_sleep,
        patch("llama_index.core.rate_limiter.time.monotonic") as mock_time,
    ):
        base = 1000.0
        rl._request_timestamps.clear()
        rl._request_timestamps.extend([base - 50, base - 40, base - 30])
        mock_time.side_effect = [base, base + 10, base + 30]
        mock_sleep.return_value = None
        rl.acquire()
        mock_sleep.assert_called()
        first_call_arg = mock_sleep.call_args_list[0][0][0]
        assert first_call_arg >= 9.0 and first_call_arg <= 11.0


def test_sliding_window_prune_removes_old_entries() -> None:
    """Entries older than 60 seconds are pruned before checking limit."""
    rl = SlidingWindowRateLimiter(requests_per_minute=2)
    now = 2000.0
    rl._request_timestamps.append(now - 70.0)
    rl._request_timestamps.append(now - 65.0)
    rl._request_timestamps.append(now - 5.0)
    with patch("llama_index.core.rate_limiter.time.monotonic", return_value=now):
        with rl._lock:
            rl._prune_request_timestamps(now)
    assert len(rl._request_timestamps) == 1


def test_sliding_window_tpm_within_limit() -> None:
    """Token usage within TPM limit should not block."""
    rl = SlidingWindowRateLimiter(tokens_per_minute=1000)
    start = time.monotonic()
    rl.acquire(num_tokens=100)
    rl.acquire(num_tokens=200)
    rl.acquire(num_tokens=300)
    elapsed = time.monotonic() - start
    assert elapsed < 1.0


def test_sliding_window_tpm_blocks_when_exceeded() -> None:
    """When token usage in window would exceed TPM, acquire blocks."""
    rl = SlidingWindowRateLimiter(tokens_per_minute=100)
    rl.acquire(num_tokens=100)

    with (
        patch("llama_index.core.rate_limiter.time.sleep") as mock_sleep,
        patch("llama_index.core.rate_limiter.time.monotonic") as mock_time,
    ):
        base = 5000.0
        rl._token_usage.clear()
        rl._token_usage.append((base - 50.0, 100.0))
        mock_time.side_effect = [base, base + 10.0]
        mock_sleep.return_value = None
        rl.acquire(num_tokens=50)
        mock_sleep.assert_called_once()
        assert mock_sleep.call_args[0][0] >= 9.0


@pytest.mark.asyncio
async def test_sliding_window_async_acquire_burst() -> None:
    rl = SlidingWindowRateLimiter(requests_per_minute=15)
    start = time.monotonic()
    for _ in range(15):
        await rl.async_acquire()
    elapsed = time.monotonic() - start
    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_sliding_window_concurrent_async() -> None:
    """Multiple concurrent async_acquire calls must all complete."""
    rl = SlidingWindowRateLimiter(requests_per_minute=100)
    results: list = []

    async def worker(n: int) -> None:
        await rl.async_acquire()
        results.append(n)

    tasks = [worker(i) for i in range(25)]
    await asyncio.gather(*tasks)
    assert len(results) == 25


def test_sliding_window_llm_sync_calls_acquire() -> None:
    """MockLLM with SlidingWindowRateLimiter should call acquire."""
    mock_limiter = MagicMock(spec=SlidingWindowRateLimiter)
    mock_limiter.acquire = MagicMock()
    mock_limiter.async_acquire = AsyncMock()
    llm = MockLLM()
    llm.rate_limiter = mock_limiter
    llm.complete("hello")
    mock_limiter.acquire.assert_called_once()


@pytest.mark.asyncio
async def test_sliding_window_llm_async_calls_async_acquire() -> None:
    mock_limiter = MagicMock(spec=SlidingWindowRateLimiter)
    mock_limiter.acquire = MagicMock()
    mock_limiter.async_acquire = AsyncMock()
    llm = MockLLM()
    llm.rate_limiter = mock_limiter
    await llm.acomplete("hello")
    mock_limiter.async_acquire.assert_called_once()


def test_sliding_window_embedding_calls_acquire() -> None:
    from llama_index.core.embeddings.mock_embed_model import MockEmbedding

    rl = SlidingWindowRateLimiter(requests_per_minute=100)
    embed = MockEmbedding(embed_dim=8, rate_limiter=rl)
    result = embed.get_text_embedding("test")
    assert len(result) == 8
