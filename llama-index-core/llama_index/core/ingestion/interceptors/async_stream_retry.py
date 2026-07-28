from __future__ import annotations

import asyncio
import random
from typing import Any, AsyncIterator, Callable, Optional, Sequence, Union


class AsyncStreamRetryInterceptor:
    """Wrap an async iterator producer and retry on transient failures.

    - Exponential backoff with jitter for exceptions and HTTP-like status codes
    - Intended for vector DB ingestion streams (batches/chunks)
    - Opt-in: import and use explicitly in pipelines
    """

    def __init__(
        self,
        max_retries: int = 3,
        backoff_base: float = 0.2,
        backoff_factor: float = 2.0,
        max_backoff: float = 5.0,
        jitter: float = 0.5,
        retry_statuses: Sequence[int] = (429, 500, 502, 503, 504),
        retry_exceptions: Sequence[type[BaseException]] = (TimeoutError, OSError),
    ) -> None:
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_factor = backoff_factor
        self.max_backoff = max_backoff
        self.jitter = jitter
        self.retry_statuses = set(retry_statuses)
        self.retry_exceptions = tuple(retry_exceptions)

    async def run(
        self,
        producer: Union[Callable[[], AsyncIterator[Any]], AsyncIterator[Any]],
    ) -> AsyncIterator[Any]:
        attempt = 0
        while True:
            try:
                ait = producer() if callable(producer) else producer
                async for item in ait:
                    yield item
                return  # completed successfully
            except Exception as exc:  # noqa: BLE001 - propagate when not retryable
                if attempt >= self.max_retries or not self._is_retryable(exc):
                    raise
                attempt += 1
                delay = self._compute_backoff(attempt, exc)
                await asyncio.sleep(delay)

    def _is_retryable(self, exc: BaseException) -> bool:
        if isinstance(exc, self.retry_exceptions):
            return True
        # HTTP-like errors: look for common attributes
        status = getattr(exc, "status", None) or getattr(exc, "code", None)
        if isinstance(status, int) and status in self.retry_statuses:
            return True
        # response-like container
        resp = getattr(exc, "response", None)
        if resp is not None:
            s = getattr(resp, "status", None) or getattr(resp, "status_code", None)
            if isinstance(s, int) and s in self.retry_statuses:
                return True
        return False

    def _retry_after(self, exc: BaseException) -> Optional[float]:
        # Honor explicit retry_after attributes when present
        ra = getattr(exc, "retry_after", None)
        try:
            return float(ra) if ra is not None else None
        except (TypeError, ValueError):
            return None

    def _compute_backoff(self, attempt: int, exc: BaseException) -> float:
        ra = self._retry_after(exc)
        base = min(self.max_backoff, self.backoff_base * (self.backoff_factor ** (attempt - 1)))
        if ra is not None:
            base = max(base, ra)
        if self.jitter > 0:
            jitter_span = base * self.jitter
            return max(0.0, base - jitter_span + random.random() * 2 * jitter_span)
        return max(0.0, base)
