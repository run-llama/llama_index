"""Rate limiters for LLM and embedding API calls."""

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

from llama_index.core.bridge.pydantic import BaseModel, Field, PrivateAttr

logger = logging.getLogger(__name__)


class BaseRateLimiter(ABC):
    """
    Abstract base class for rate limiters.

    All rate limiter implementations must inherit from this class and
    implement :meth:`acquire` (synchronous) and :meth:`async_acquire`
    (asynchronous). This allows swapping in alternative strategies
    (e.g. distributed rate limiting via Redis) without changing the
    calling code in ``BaseLLM`` or ``BaseEmbedding``.
    """

    @abstractmethod
    def acquire(self, num_tokens: int = 0) -> None:
        """
        Block until one request is allowed (synchronous).

        Args:
            num_tokens: Estimated token count for this request.
                Implementations may ignore this if they only track
                request counts.

        """

    @abstractmethod
    async def async_acquire(self, num_tokens: int = 0) -> None:
        """
        Wait until one request is allowed (asynchronous).

        Args:
            num_tokens: Estimated token count for this request.
                Implementations may ignore this if they only track
                request counts.

        """


class TokenBucketRateLimiter(BaseRateLimiter, BaseModel):
    """
    Token-bucket rate limiter for controlling API request throughput.

    Supports both requests-per-minute (RPM) and tokens-per-minute (TPM)
    limiting. Instances can be shared across multiple LLM and embedding
    objects that hit the same API endpoint, so a single budget is enforced
    globally.

    A token-bucket allows an initial burst up to the configured limit,
    then smoothly throttles to the sustained rate. This matches the
    behaviour of most LLM provider rate limits.

    Args:
        requests_per_minute: Maximum requests allowed per minute.
            ``None`` disables request-rate limiting.
        tokens_per_minute: Maximum tokens allowed per minute.
            ``None`` disables token-rate limiting.

    Examples:
        .. code-block:: python

            from llama_index.core.rate_limiter import TokenBucketRateLimiter

            # Share a single limiter across LLM and embedding instances
            groq_limiter = TokenBucketRateLimiter(requests_per_minute=30)
            llm = SomeLLM(rate_limiter=groq_limiter)
            embed = SomeEmbedding(rate_limiter=groq_limiter)

    """

    requests_per_minute: Optional[float] = Field(
        default=None,
        description="Maximum number of requests per minute.",
        gt=0,
    )
    tokens_per_minute: Optional[float] = Field(
        default=None,
        description="Maximum number of tokens per minute.",
        gt=0,
    )

    _request_tokens: float = PrivateAttr(default=0.0)
    _request_max_tokens: float = PrivateAttr(default=0.0)
    _request_refill_rate: float = PrivateAttr(default=0.0)
    _token_tokens: float = PrivateAttr(default=0.0)
    _token_max_tokens: float = PrivateAttr(default=0.0)
    _token_refill_rate: float = PrivateAttr(default=0.0)
    _last_refill_time: float = PrivateAttr(default=0.0)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        now = time.monotonic()
        self._last_refill_time = now

        if self.requests_per_minute is not None:
            self._request_max_tokens = self.requests_per_minute
            self._request_tokens = self.requests_per_minute
            self._request_refill_rate = self.requests_per_minute / 60.0

        if self.tokens_per_minute is not None:
            self._token_max_tokens = self.tokens_per_minute
            self._token_tokens = self.tokens_per_minute
            self._token_refill_rate = self.tokens_per_minute / 60.0

    def _refill(self) -> None:
        """
        Refill token buckets based on elapsed time.

        Must be called while holding ``_lock``.
        """
        now = time.monotonic()
        elapsed = now - self._last_refill_time
        self._last_refill_time = now

        if self.requests_per_minute is not None:
            self._request_tokens = min(
                self._request_max_tokens,
                self._request_tokens + elapsed * self._request_refill_rate,
            )
        if self.tokens_per_minute is not None:
            self._token_tokens = min(
                self._token_max_tokens,
                self._token_tokens + elapsed * self._token_refill_rate,
            )

    def _wait_time(self, num_tokens: int = 0) -> float:
        """
        Return seconds to wait before the next request is allowed.

        Must be called while holding ``_lock`` and after ``_refill()``.
        """
        wait = 0.0
        if self.requests_per_minute is not None and self._request_tokens < 1.0:
            wait = max(
                wait,
                (1.0 - self._request_tokens) / self._request_refill_rate,
            )
        if (
            self.tokens_per_minute is not None
            and num_tokens > 0
            and self._token_tokens < num_tokens
        ):
            wait = max(
                wait,
                (num_tokens - self._token_tokens) / self._token_refill_rate,
            )
        return wait

    def _consume(self, num_tokens: int = 0) -> None:
        """
        Consume one request token and *num_tokens* LLM tokens.

        Must be called while holding ``_lock``.
        """
        if self.requests_per_minute is not None:
            self._request_tokens -= 1.0
        if self.tokens_per_minute is not None and num_tokens > 0:
            self._token_tokens -= num_tokens

    def acquire(self, num_tokens: int = 0) -> None:
        """
        Block until one request is allowed (synchronous).

        Args:
            num_tokens: Estimated token count for this request.  Only
                consulted when ``tokens_per_minute`` is configured.

        """
        while True:
            with self._lock:
                self._refill()
                wait = self._wait_time(num_tokens)
                if wait <= 0:
                    self._consume(num_tokens)
                    return
            time.sleep(wait)

    async def async_acquire(self, num_tokens: int = 0) -> None:
        """
        Wait until one request is allowed (asynchronous).

        Args:
            num_tokens: Estimated token count for this request.  Only
                consulted when ``tokens_per_minute`` is configured.

        """
        while True:
            with self._lock:
                self._refill()
                wait = self._wait_time(num_tokens)
                if wait <= 0:
                    self._consume(num_tokens)
                    return
            await asyncio.sleep(wait)


# Backwards-compatible alias
RateLimiter = TokenBucketRateLimiter
