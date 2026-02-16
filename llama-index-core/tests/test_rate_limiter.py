"""Tests for the token-bucket rate limiter."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llama_index.core.base.llms.types import (
    ChatMessage,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.rate_limiter import RateLimiter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubLLM(CustomLLM):
    """Minimal LLM for testing rate-limiter integration."""

    __test__ = False

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata()

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: object
    ) -> CompletionResponse:
        return CompletionResponse(text="ok")

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: object
    ) -> CompletionResponseGen:
        def gen() -> CompletionResponseGen:
            yield CompletionResponse(text="ok", delta="ok")

        return gen()


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
async def test_aacquire_burst_within_limit() -> None:
    rl = RateLimiter(requests_per_minute=10)
    start = time.monotonic()
    for _ in range(10):
        await rl.aacquire()
    elapsed = time.monotonic() - start
    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_aacquire_tpm_limiting() -> None:
    """TPM limiting should throttle based on token count."""
    rl = RateLimiter(tokens_per_minute=100)
    await rl.aacquire(num_tokens=100)

    with (
        patch("llama_index.core.rate_limiter.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        patch("llama_index.core.rate_limiter.time.monotonic") as mock_time,
    ):
        base = 1000.0
        # Align internal clock with mock timeline
        rl._last_refill_time = base
        rl._token_tokens = 0.0
        # First monotonic(): no elapsed time, need 50 tokens but 0 available -> sleep
        # Second monotonic(): 60s elapsed -> refills 100 tokens -> acquire succeeds
        mock_time.side_effect = [base, base + 60.0]
        await rl.aacquire(num_tokens=50)
        mock_sleep.assert_called_once()


@pytest.mark.asyncio
async def test_concurrent_async_rate_limiting() -> None:
    """Multiple concurrent aacquire calls must all complete."""
    rl = RateLimiter(requests_per_minute=600)
    results: list = []

    async def worker(n: int) -> None:
        await rl.aacquire()
        results.append(n)

    tasks = [worker(i) for i in range(20)]
    await asyncio.gather(*tasks)
    assert len(results) == 20


# ---------------------------------------------------------------------------
# LLM integration tests
# ---------------------------------------------------------------------------


def test_llm_sync_chat_calls_acquire() -> None:
    mock_limiter = MagicMock()
    llm = _StubLLM(rate_limiter=mock_limiter)
    llm.chat([ChatMessage(role="user", content="hello")])
    # chat() triggers callback; internally calls complete() which also triggers
    assert mock_limiter.acquire.call_count >= 1


@pytest.mark.asyncio
async def test_llm_async_chat_calls_aacquire() -> None:
    mock_limiter = MagicMock()
    mock_limiter.aacquire = AsyncMock()
    llm = _StubLLM(rate_limiter=mock_limiter)
    await llm.achat([ChatMessage(role="user", content="hello")])
    mock_limiter.aacquire.assert_called_once()


def test_llm_sync_complete_calls_acquire() -> None:
    mock_limiter = MagicMock()
    llm = _StubLLM(rate_limiter=mock_limiter)
    llm.complete("hello")
    mock_limiter.acquire.assert_called_once()


@pytest.mark.asyncio
async def test_llm_async_complete_calls_aacquire() -> None:
    mock_limiter = MagicMock()
    mock_limiter.aacquire = AsyncMock()
    llm = _StubLLM(rate_limiter=mock_limiter)
    await llm.acomplete("hello")
    mock_limiter.aacquire.assert_called_once()


def test_llm_without_rate_limiter_works() -> None:
    llm = _StubLLM()
    assert llm.rate_limiter is None
    response = llm.chat([ChatMessage(role="user", content="hello")])
    assert response.message.content == "ok"


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
    llm = _StubLLM(rate_limiter=rl)
    embed = MockEmbedding(embed_dim=8, rate_limiter=rl)
    assert llm.rate_limiter is embed.rate_limiter
