import logging
from typing import Optional, Tuple, Callable, Any

import openai
from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from tenacity import (
    wait_random_exponential,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
    retry,
    before_sleep_log,
)
from tenacity.stop import stop_base, stop_after_delay

DEFAULT_UPSTAGE_API_BASE = "https://api.upstage.ai/v1/solar"

logger = logging.getLogger(__name__)


def resolve_upstage_credentials(
    api_key: Optional[str] = None, api_base: Optional[str] = None
) -> Tuple[Optional[str], str]:
    """
    Resolve Upstage credentials.

    The order of precedence is:
    1. param
    2. env
    4. default
    """
    # resolve from param or env
    api_key = get_from_param_or_env("api_key", api_key, "UPSTAGE_API_KEY", "")
    api_base = get_from_param_or_env("api_base", api_base, "UPSTAGE_API_BASE", "")

    final_api_key = api_key or ""
    final_api_base = api_base or DEFAULT_UPSTAGE_API_BASE

    return final_api_key, str(final_api_base)


def create_retry_decorator(
    max_retries: int,
    random_exponential: bool = False,
    stop_after_delay_seconds: Optional[float] = None,
    min_seconds: float = 4,
    max_seconds: float = 10,
) -> Callable[[Any], Any]:
    wait_strategy = (
        wait_random_exponential(min=min_seconds, max=max_seconds)
        if random_exponential
        else wait_exponential(multiplier=1, min=min_seconds, max=max_seconds)
    )

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
