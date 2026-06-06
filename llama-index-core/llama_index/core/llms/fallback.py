"""Fallback LLM wrapper for automatic provider failover."""

import logging
from typing import Any, List, Optional, Sequence

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.core.llms.llm import LLM
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error classification helpers
# ---------------------------------------------------------------------------

def _is_transient_error(exception: Exception) -> bool:
    """
    Determine whether an error is transient (worth retrying) or permanent.

    Transient errors include:
    - Timeout errors
    - Connection errors (ConnectionError, ConnectionResetError, etc.)
    - HTTP 429 (Too Many Requests)
    - HTTP 5xx (Server errors)
    - RateLimitError variants from common SDKs

    Permanent errors include:
    - HTTP 4xx (except 429) — auth, bad request, not found
    - Model validation / bad request errors
    - Context length exceeded (explicitly permanent)
    """
    import importlib

    error_str = str(exception).lower()
    error_repr = repr(exception).lower()

    # --- Category-tag style errors (used by many provider SDKs) ---
    transient_keywords = [
        "timeout",
        "timed out",
        "connection",
        "reset by peer",
        "broken pipe",
        "service unavailable",
        "server error",
        "internal server error",
        "bad gateway",
        "gateway timeout",
        "too many requests",
        "rate limit",
        "rate exceeded",
        "overloaded",
        "try again",
        "retry",
    ]
    permanent_keywords = [
        "invalid api key",
        "invalid authentication",
        "unauthorized",
        "forbidden",
        "not found",
        "context length exceeded",
        "context_length_exceeded",
        "invalid request",
        "bad request",
        "content filter",
        "content policy",
        "safety",
        "moderation",
        "invalid model",
    ]

    # Omit the explicit str/repr checks and rely on dynamic attr probing
    # Check for transient keywords in the error message / repr
    combined = f"{error_str} {error_repr}"
    for keyword in transient_keywords:
        if keyword in combined:
            return True
    for keyword in permanent_keywords:
        if keyword in combined:
            return False

    # --- HTTP status code checks on exception attributes ---
    for attr_name in ("status_code", "http_status", "status"):
        try:
            code = int(getattr(exception, attr_name, 0))
            if code == 429 or 500 <= code < 600:
                return True
            if 400 <= code < 500:
                return False
        except (TypeError, ValueError):
            pass

    # --- Check known SDK error type names ---
    error_cls_name = type(exception).__qualname__
    error_module = type(exception).__module__

    # Transient error classes (by naming convention)
    transient_class_patterns = [
        "timeout",
        "connectionerror",
        "connection",
        "ratelimiterror",
        "ratelimit",
        "serviceunavailable",
        "internalservererror",
        "apiconnectionerror",
        "apitimeouterror",
        "servererror",
    ]
    permanent_class_patterns = [
        "authenticationerror",
        "permissiondeniederror",
        "invalidrequesterror",
        "badrequesterror",
        "invalidrequesterror",
        "contentfiltererror",
        "safetyerror",
        "unprocessableentityerror",
    ]

    for pattern in transient_class_patterns:
        if pattern in error_cls_name.lower():
            return True
    for pattern in permanent_class_patterns:
        if pattern in error_cls_name.lower():
            return False

    # Default: treat unknown errors as transient (safer to try next provider)
    return True


def _is_retryable_error(exception: Exception) -> bool:
    """Alias for _is_transient_error — kept for backward compatibility."""
    return _is_transient_error(exception)


# ---------------------------------------------------------------------------
# FallbackLLM
# ---------------------------------------------------------------------------


class FallbackLLM(LLM):
    """
    An LLM wrapper that provides automatic failover across multiple providers.

    When a call to the primary LLM fails with a transient error, the next
    provider in the list is tried automatically.  Permanent errors (auth,
    bad requests, etc.) are surfaced immediately.

    Args:
        llms: List of LLM instances in priority order.
        retry_attempts: Number of retries per provider before falling back
            to the next provider (default: 1).
        timeout: Optional timeout in seconds for each individual call.

    Examples:
        ```python
        from llama_index.llms.openai import OpenAI
        from llama_index.llms.anthropic import Anthropic
        from llama_index.core.llms.fallback import FallbackLLM

        openai_llm = OpenAI(model="gpt-4o")
        anthropic_llm = Anthropic(model="claude-sonnet-4-20250514")

        llm = FallbackLLM(
            llms=[openai_llm, anthropic_llm],
            retry_attempts=2,
        )
        response = llm.complete("Hello, world!")
        ```
    """

    llms: List[LLM]
    retry_attempts: int = 1
    timeout: Optional[float] = None

    def __init__(
        self,
        llms: List[LLM],
        retry_attempts: int = 1,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        if not llms:
            raise ValueError("At least one LLM must be provided.")
        super().__init__(
            llms=llms,
            retry_attempts=retry_attempts,
            timeout=timeout,
            **kwargs,
        )

    @property
    def metadata(self) -> LLMMetadata:
        """Return metadata from the first (primary) LLM in the list."""
        primary = self.llms[0]
        # Use primary's metadata but override context_window / num_output
        # with safe defaults if the primary's metadata is not set.
        primary_meta = primary.metadata
        return LLMMetadata(
            context_window=primary_meta.context_window or DEFAULT_CONTEXT_WINDOW,
            num_output=primary_meta.num_output or DEFAULT_NUM_OUTPUTS,
            is_chat_model=primary_meta.is_chat_model,
            is_function_calling_model=primary_meta.is_function_calling_model,
            model_name=f"fallback[{','.join(llm.metadata.model_name for llm in self.llms)}]",
            system_role=primary_meta.system_role,
        )

    @classmethod
    def class_name(cls) -> str:
        return "fallback_llm"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_all(
        self,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Iterate through providers, calling *method_name* on each.

        Retries *retry_attempts* times per provider before falling back.
        Raises the last error if all providers are exhausted.
        """
        last_error: Optional[Exception] = None

        for idx, llm in enumerate(self.llms):
            llm_name = llm.metadata.model_name if hasattr(llm, "metadata") else str(llm)  # type: ignore[assignment]
            for attempt in range(self.retry_attempts):
                try:
                    method = getattr(llm, method_name)
                    result = method(*args, **kwargs)
                    if idx > 0:
                        _logger.info(
                            "FallbackLLM: successfully used fallback provider "
                            "'%s' (index %d) after primary failure.",
                            llm_name,
                            idx,
                        )
                    return result
                except Exception as exc:
                    last_error = exc
                    if not _is_transient_error(exc):
                        _logger.error(
                            "FallbackLLM: permanent error from provider "
                            "'%s' (index %d) — not retrying: %s",
                            llm_name,
                            idx,
                            exc,
                        )
                        raise
                    retries_left = self.retry_attempts - attempt - 1
                    providers_left = len(self.llms) - idx - 1
                    if retries_left > 0:
                        _logger.warning(
                            "FallbackLLM: transient error from provider "
                            "'%s' (attempt %d/%d, %d retries left): %s",
                            llm_name,
                            attempt + 1,
                            self.retry_attempts,
                            retries_left,
                            exc,
                        )
                    elif providers_left > 0:
                        _logger.warning(
                            "FallbackLLM: exhausted retries for provider "
                            "'%s', falling back to next provider (%d remaining): %s",
                            llm_name,
                            providers_left,
                            exc,
                        )
                    else:
                        _logger.error(
                            "FallbackLLM: all providers exhausted.  Last error: %s",
                            exc,
                        )

        assert last_error is not None
        raise RuntimeError(
            f"FallbackLLM: all {len(self.llms)} provider(s) failed.  "
            f"Last error: {last_error!r}"
        ) from last_error

    async def _atry_all(
        self,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Async version of _try_all."""
        last_error: Optional[Exception] = None

        for idx, llm in enumerate(self.llms):
            llm_name = llm.metadata.model_name if hasattr(llm, "metadata") else str(llm)  # type: ignore[assignment]
            for attempt in range(self.retry_attempts):
                try:
                    method = getattr(llm, method_name)
                    result = method(*args, **kwargs)
                    # Await if the result is a coroutine
                    import asyncio

                    if asyncio.iscoroutine(result):
                        result = await result
                    if idx > 0:
                        _logger.info(
                            "FallbackLLM: successfully used fallback provider "
                            "'%s' (index %d) after primary failure.",
                            llm_name,
                            idx,
                        )
                    return result
                except Exception as exc:
                    last_error = exc
                    if not _is_transient_error(exc):
                        _logger.error(
                            "FallbackLLM: permanent error from provider "
                            "'%s' (index %d) — not retrying: %s",
                            llm_name,
                            idx,
                            exc,
                        )
                        raise
                    retries_left = self.retry_attempts - attempt - 1
                    providers_left = len(self.llms) - idx - 1
                    if retries_left > 0:
                        _logger.warning(
                            "FallbackLLM: transient error from provider "
                            "'%s' (attempt %d/%d, %d retries left): %s",
                            llm_name,
                            attempt + 1,
                            self.retry_attempts,
                            retries_left,
                            exc,
                        )
                    elif providers_left > 0:
                        _logger.warning(
                            "FallbackLLM: exhausted retries for provider "
                            "'%s', falling back to next provider (%d remaining): %s",
                            llm_name,
                            providers_left,
                            exc,
                        )
                    else:
                        _logger.error(
                            "FallbackLLM: all providers exhausted.  Last error: %s",
                            exc,
                        )

        assert last_error is not None
        raise RuntimeError(
            f"FallbackLLM: all {len(self.llms)} provider(s) failed.  "
            f"Last error: {last_error!r}"
        ) from last_error

    def _try_all_generator(
        self,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Generator version — for streaming endpoints.

        Because generators are lazy, we catch errors only when the consumer
        starts iterating.  We wrap each generator in a helper that catches
        errors on first yield.
        """
        last_error: Optional[Exception] = None

        for idx, llm in enumerate(self.llms):
            llm_name = llm.metadata.model_name if hasattr(llm, "metadata") else str(llm)  # type: ignore[assignment]
            for attempt in range(self.retry_attempts):
                try:
                    method = getattr(llm, method_name)
                    gen = method(*args, **kwargs)
                    # Try to yield the first item to catch errors eagerly
                    first = next(gen)
                    if idx > 0:
                        _logger.info(
                            "FallbackLLM: successfully used fallback provider "
                            "'%s' (index %d) for streaming after primary failure.",
                            llm_name,
                            idx,
                        )

                    # Re-wrap: yield the first item, then delegate to the rest
                    def _regen(first_val: Any, gen_rest: Any) -> Any:
                        yield first_val
                        yield from gen_rest

                    return _regen(first, gen)
                except StopIteration:
                    # Empty generator — treat as success
                    if idx > 0:
                        _logger.info(
                            "FallbackLLM: successfully used fallback provider "
                            "'%s' (index %d) for streaming (empty).",
                            llm_name,
                            idx,
                        )
                    return iter([])
                except Exception as exc:
                    last_error = exc
                    if not _is_transient_error(exc):
                        _logger.error(
                            "FallbackLLM: permanent error from provider "
                            "'%s' (index %d) during streaming — not retrying: %s",
                            llm_name,
                            idx,
                            exc,
                        )
                        raise
                    retries_left = self.retry_attempts - attempt - 1
                    providers_left = len(self.llms) - idx - 1
                    if retries_left > 0:
                        _logger.warning(
                            "FallbackLLM: transient error from provider "
                            "'%s' during streaming (attempt %d/%d, %d retries left): %s",
                            llm_name,
                            attempt + 1,
                            self.retry_attempts,
                            retries_left,
                            exc,
                        )
                    elif providers_left > 0:
                        _logger.warning(
                            "FallbackLLM: exhausted retries for provider "
                            "'%s' during streaming, falling back to next provider (%d remaining): %s",
                            llm_name,
                            providers_left,
                            exc,
                        )
                    else:
                        _logger.error(
                            "FallbackLLM: all providers exhausted during streaming.  Last error: %s",
                            exc,
                        )

        assert last_error is not None
        raise RuntimeError(
            f"FallbackLLM: all {len(self.llms)} provider(s) failed during streaming.  "
            f"Last error: {last_error!r}"
        ) from last_error

    async def _atry_all_generator(
        self,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Async generator version for async streaming endpoints."""
        import asyncio

        last_error: Optional[Exception] = None

        for idx, llm in enumerate(self.llms):
            llm_name = llm.metadata.model_name if hasattr(llm, "metadata") else str(llm)  # type: ignore[assignment]
            for attempt in range(self.retry_attempts):
                try:
                    method = getattr(llm, method_name)
                    agen = method(*args, **kwargs)
                    if asyncio.iscoroutine(agen):
                        agen = await agen
                    first = await agen.__anext__()
                    if idx > 0:
                        _logger.info(
                            "FallbackLLM: successfully used fallback provider "
                            "'%s' (index %d) for async streaming after primary failure.",
                            llm_name,
                            idx,
                        )

                    async def _aregen(first_val: Any, gen_rest: Any) -> Any:
                        yield first_val
                        async for item in gen_rest:
                            yield item

                    return _aregen(first, agen)
                except StopAsyncIteration:
                    if idx > 0:
                        _logger.info(
                            "FallbackLLM: successfully used fallback provider "
                            "'%s' (index %d) for async streaming (empty).",
                            llm_name,
                            idx,
                        )

                    async def _empty() -> Any:
                        if False:
                            yield  # pragma: no cover

                    return _empty()
                except Exception as exc:
                    last_error = exc
                    if not _is_transient_error(exc):
                        _logger.error(
                            "FallbackLLM: permanent error from provider "
                            "'%s' (index %d) during async streaming — not retrying: %s",
                            llm_name,
                            idx,
                            exc,
                        )
                        raise
                    retries_left = self.retry_attempts - attempt - 1
                    providers_left = len(self.llms) - idx - 1
                    if retries_left > 0:
                        _logger.warning(
                            "FallbackLLM: transient error from provider "
                            "'%s' during async streaming (attempt %d/%d, %d retries left): %s",
                            llm_name,
                            attempt + 1,
                            self.retry_attempts,
                            retries_left,
                            exc,
                        )
                    elif providers_left > 0:
                        _logger.warning(
                            "FallbackLLM: exhausted retries for provider "
                            "'%s' during async streaming, falling back to next provider (%d remaining): %s",
                            llm_name,
                            providers_left,
                            exc,
                        )
                    else:
                        _logger.error(
                            "FallbackLLM: all providers exhausted during async streaming.  Last error: %s",
                            exc,
                        )

        assert last_error is not None
        raise RuntimeError(
            f"FallbackLLM: all {len(self.llms)} provider(s) failed during async streaming.  "
            f"Last error: {last_error!r}"
        ) from last_error

    # ------------------------------------------------------------------
    # Synchronous endpoints
    # ------------------------------------------------------------------

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return self._try_all("chat", messages, **kwargs)

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return self._try_all("complete", prompt, formatted=formatted, **kwargs)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        return self._try_all_generator("stream_chat", messages, **kwargs)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        return self._try_all_generator(
            "stream_complete", prompt, formatted=formatted, **kwargs
        )

    # ------------------------------------------------------------------
    # Async endpoints
    # ------------------------------------------------------------------

    @llm_chat_callback()
    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        return await self._atry_all("achat", messages, **kwargs)

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return await self._atry_all(
            "acomplete", prompt, formatted=formatted, **kwargs
        )

    @llm_chat_callback()
    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        return await self._atry_all_generator("astream_chat", messages, **kwargs)

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        return await self._atry_all_generator(
            "astream_complete", prompt, formatted=formatted, **kwargs
        )
