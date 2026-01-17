from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any, Callable

import httpx
from google.genai import errors
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_random_exponential,
)
from tenacity.stop import stop_base


if TYPE_CHECKING:
    from llama_index.llms.google_genai.base import GoogleGenAI


logger = logging.getLogger(__name__)


def _should_retry(exception: BaseException) -> bool:
    if isinstance(exception, errors.ClientError):
        return exception.status in (429, 408)
    return False


def _create_retry_decorator(max_retries: int) -> Callable[[Any], Any]:
    wait_strategy = wait_random_exponential(min=1, max=20)
    stop_strategy: stop_base = stop_after_attempt(max_retries) | stop_after_delay(60)
    return retry(
        reraise=True,
        stop=stop_strategy,
        wait=wait_strategy,
        retry=(
            retry_if_exception_type(
                (errors.ServerError, httpx.ConnectError, httpx.ConnectTimeout)
            )
            | retry_if_exception(_should_retry)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def llm_retry_decorator(f: Callable[..., Any]) -> Callable[..., Any]:
    """Retry decorator for LLM calls using this instance's ``max_retries``."""

    @functools.wraps(f)
    def wrapper(self: "GoogleGenAI", *args: Any, **kwargs: Any) -> Any:
        max_retries = getattr(self, "max_retries", 0)
        if max_retries <= 0:
            return f(self, *args, **kwargs)

        retry_decorator = _create_retry_decorator(max_retries=max_retries)
        return retry_decorator(f)(self, *args, **kwargs)

    return wrapper
