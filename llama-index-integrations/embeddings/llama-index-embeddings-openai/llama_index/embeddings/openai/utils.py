import logging
import os
from typing import Any, Callable, Optional, Tuple, Union

from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from tenacity import (
    RetryCallState,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
    wait_random_exponential,
)
from tenacity.stop import stop_base
from tenacity.wait import wait_base

import openai
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall

DEFAULT_OPENAI_API_BASE = "https://api.openai.com/v1"
DEFAULT_OPENAI_API_VERSION = ""

MISSING_API_KEY_ERROR_MESSAGE = """No API key found for OpenAI.
Please set either the OPENAI_API_KEY environment variable or \
openai.api_key prior to initialization.
API keys can be found or created at \
https://platform.openai.com/account/api-keys
"""

logger = logging.getLogger(__name__)

OpenAIToolCall = Union[ChatCompletionMessageToolCall, ChoiceDeltaToolCall]

# Maximum wait time (seconds) to accept from a Retry-After header.
# Prevents a misbehaving server from stalling the client indefinitely.
_MAX_RETRY_AFTER_SECONDS = 120.0


class _WaitRetryAfter(wait_base):
    """
    Wait strategy that respects the Retry-After header on RateLimitError.

    When the last exception is an ``openai.RateLimitError`` whose HTTP response
    contains a ``Retry-After`` header, the wait time is taken from that header
    (capped at ``_MAX_RETRY_AFTER_SECONDS``).  For all other exceptions the
    ``fallback`` strategy decides the sleep duration.
    """

    def __init__(self, fallback: wait_base) -> None:
        self.fallback = fallback

    def __call__(self, retry_state: RetryCallState) -> float:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        if isinstance(exc, openai.RateLimitError):
            retry_after = _parse_retry_after(exc)
            if retry_after is not None:
                return min(retry_after, _MAX_RETRY_AFTER_SECONDS)
        return self.fallback(retry_state)


def _parse_retry_after(exc: openai.RateLimitError) -> Optional[float]:
    """
    Extract the Retry-After value (in seconds) from a RateLimitError.

    Returns ``None`` when the header is missing or cannot be parsed.
    """
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after")  # httpx.Headers is case-insensitive
    if raw is None:
        return None
    try:
        value = float(raw)
    except (ValueError, TypeError):
        return None
    if value < 0:
        return None
    return value


def create_retry_decorator(
    max_retries: int,
    random_exponential: bool = False,
    stop_after_delay_seconds: Optional[float] = None,
    min_seconds: float = 4,
    max_seconds: float = 10,
) -> Callable[[Any], Any]:
    fallback = (
        wait_random_exponential(min=min_seconds, max=max_seconds)
        if random_exponential
        else wait_exponential(multiplier=1, min=min_seconds, max=max_seconds)
    )
    wait_strategy = _WaitRetryAfter(fallback)

    stop_strategy: stop_base = stop_after_attempt(max_retries)
    if stop_after_delay_seconds is not None:
        stop_strategy = stop_strategy | stop_after_delay(stop_after_delay_seconds)

    return retry(
        reraise=True,
        stop=stop_strategy,
        wait=wait_strategy,
        retry=(
            retry_if_exception_type(
                (
                    openai.APIConnectionError,
                    openai.APITimeoutError,
                    openai.RateLimitError,
                    openai.InternalServerError,
                )
            )
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def resolve_openai_credentials(
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
) -> Tuple[Optional[str], str, str]:
    """
    "Resolve OpenAI credentials.

    The order of precedence is:
    1. param
    2. env
    3. openai module
    4. default
    """
    # resolve from param or env
    api_key = get_from_param_or_env("api_key", api_key, "OPENAI_API_KEY", "")
    api_base = get_from_param_or_env("api_base", api_base, "OPENAI_API_BASE", "")
    api_version = get_from_param_or_env(
        "api_version", api_version, "OPENAI_API_VERSION", ""
    )

    # resolve from openai module or default
    final_api_key = api_key or openai.api_key or ""
    final_api_base = api_base or openai.base_url or DEFAULT_OPENAI_API_BASE
    final_api_version = api_version or openai.api_version or DEFAULT_OPENAI_API_VERSION

    return final_api_key, str(final_api_base), final_api_version


def validate_openai_api_key(api_key: Optional[str] = None) -> None:
    openai_api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    if not openai_api_key:
        raise ValueError(MISSING_API_KEY_ERROR_MESSAGE)
