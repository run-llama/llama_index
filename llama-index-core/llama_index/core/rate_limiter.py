"""Rate limiters for LLM and embedding API calls."""

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Optional, Tuple

from llama_index.core.bridge.pydantic import BaseModel, Field, PrivateAttr

logger = logging.getLogger(__name__)

# Sliding window duration in seconds (one minute)
_SLIDING_WINDOW_SECONDS = 60.0


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


class SlidingWindowRateLimiter(BaseRateLimiter, BaseModel):
    """
    Sliding-window rate limiter for strict per-minute caps.

    Unlike the token-bucket limiter, this implementation enforces a strict
    sliding window: at any moment, only the requests (or tokens) that fall
    within the last 60 seconds count toward the limit. There is no burst
    allowance at window boundaries, which can be required by APIs that
    specify hard limits per rolling minute.

    Supports both requests-per-minute (RPM) and tokens-per-minute (TPM)
    limiting, with optional burst headroom to better match provider
    semantics. Instances can be shared across multiple LLM and embedding
    objects that hit the same API endpoint.

    Args:
        requests_per_minute: Maximum requests allowed in any sliding
            one-minute window. ``None`` disables request-rate limiting.
        request_burst: Additional requests allowed as burst capacity
            within the sliding window. Defaults to 0 (strict cap).
        tokens_per_minute: Maximum tokens allowed in any sliding
            one-minute window. ``None`` disables token-rate limiting.
        token_burst: Additional tokens allowed as burst capacity
            within the sliding window. Defaults to 0 (strict cap).

    Raises:
        ValueError: If both ``requests_per_minute`` and ``tokens_per_minute``
            are ``None``, or if either is zero or negative.

    Examples:
        .. code-block:: python

            from llama_index.core.rate_limiter import SlidingWindowRateLimiter

            limiter = SlidingWindowRateLimiter(requests_per_minute=60)
            llm = SomeLLM(rate_limiter=limiter)
    """

    requests_per_minute: Optional[float] = Field(
        default=None,
        description="Maximum number of requests in any sliding one-minute window.",
        gt=0,
    )
    request_burst: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Additional requests allowed as burst capacity within the sliding window. "
            "Set to 0 for a strict cap."
        ),
    )
    tokens_per_minute: Optional[float] = Field(
        default=None,
        description="Maximum number of tokens in any sliding one-minute window.",
        gt=0,
    )
    token_burst: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Additional tokens allowed as burst capacity within the sliding window. "
            "Set to 0 for a strict cap."
        ),
    )

    _request_timestamps: Deque[float] = PrivateAttr(default_factory=deque)
    _token_usage: Deque[Tuple[float, float]] = PrivateAttr(default_factory=deque)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if (
            self.requests_per_minute is None
            and self.tokens_per_minute is None
        ):
            raise ValueError(
                "At least one of requests_per_minute or tokens_per_minute must be set."
            )

    def _prune_request_timestamps(self, now: float) -> None:
        """Remove request timestamps outside the sliding window. Hold _lock."""
        while self._request_timestamps and self._request_timestamps[0] < now - _SLIDING_WINDOW_SECONDS:
            self._request_timestamps.popleft()

    def _prune_token_usage(self, now: float) -> None:
        """Remove token usage entries outside the sliding window. Hold _lock."""
        while self._token_usage and self._token_usage[0][0] < now - _SLIDING_WINDOW_SECONDS:
            self._token_usage.popleft()

    def _current_token_usage(self) -> float:
        """Sum of tokens currently in the sliding window. Hold _lock."""
        return sum(tokens for _ts, tokens in self._token_usage)

    def _wait_time(self, now: float, num_tokens: int = 0) -> float:
        """
        Return seconds to wait before the next request is allowed.

        Must be called while holding ``_lock`` and after pruning.
        """
        wait = 0.0
        if self.requests_per_minute is not None:
            # Allow optional burst headroom in addition to the strict window cap.
            allowed_requests = self.requests_per_minute + self.request_burst
            if len(self._request_timestamps) >= allowed_requests:
                # Wait until the oldest request exits the window
                wait = max(
                    wait,
                    self._request_timestamps[0] + _SLIDING_WINDOW_SECONDS - now,
                )
        if self.tokens_per_minute is not None and num_tokens > 0:
            current = self._current_token_usage()
            allowed_tokens = self.tokens_per_minute + self.token_burst
            if current + num_tokens > allowed_tokens:
                # Must wait until enough token usage expires
                if not self._token_usage:
                    return wait
                # How long until we can fit num_tokens (current usage must drop)
                needed = current + num_tokens - allowed_tokens
                # Expire from oldest; approximate wait from oldest entry
                remaining = needed
                for ts, tokens in self._token_usage:
                    remaining -= tokens
                    if remaining <= 0:
                        wait = max(
                            wait,
                            ts + _SLIDING_WINDOW_SECONDS - now,
                        )
                        break
        return wait

    def _record_usage(self, now: float, num_tokens: int = 0) -> None:
        """Record one request and optional token usage. Hold _lock."""
        if self.requests_per_minute is not None:
            self._request_timestamps.append(now)
        if self.tokens_per_minute is not None and num_tokens > 0:
            self._token_usage.append((now, float(num_tokens)))

    def acquire(self, num_tokens: int = 0) -> None:
        """
        Block until one request is allowed (synchronous).

        Args:
            num_tokens: Estimated token count for this request.  Only
                consulted when ``tokens_per_minute`` is configured.

        """
        while True:
            now = time.monotonic()
            with self._lock:
                self._prune_request_timestamps(now)
                self._prune_token_usage(now)
                wait = self._wait_time(now, num_tokens)
                if wait <= 0:
                    self._record_usage(now, num_tokens)
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
            now = time.monotonic()
            with self._lock:
                self._prune_request_timestamps(now)
                self._prune_token_usage(now)
                wait = self._wait_time(now, num_tokens)
                if wait <= 0:
                    self._record_usage(now, num_tokens)
                    return
            await asyncio.sleep(wait)


# Backwards-compatible alias
RateLimiter = TokenBucketRateLimiter
