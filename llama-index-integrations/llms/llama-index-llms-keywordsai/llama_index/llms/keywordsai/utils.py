import logging
import os
from typing import Any, Callable, Optional, Union, Tuple
from functools import lru_cache

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
    wait_random_exponential,
)
from tenacity.stop import stop_base
from llama_index.core.base.llms.generic_utils import get_from_param_or_env
import openai
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall

DEFAULT_KEYWORDSAI_API_TYPE = "keywords_ai"
DEFAULT_KEYWORDSAI_API_BASE = "https://api.keywordsai.co/api/"
DEFAULT_KEYWORDSAI_API_VERSION = ""

MISSING_API_KEY_ERROR_MESSAGE = """No API key found for KeywordsAI.
Please set either the KEYWORDSAI_API_KEY environment variable.
"""

logger = logging.getLogger(__name__)

KeywordsAIToolCall = Union[ChatCompletionMessageToolCall, ChoiceDeltaToolCall]


@lru_cache
def get_keywords_models():
    """Get available models from KeywordsAI API.

    Returns:
        List of model configurations with pricing and provider info.

    Raises:
        Exception if API call fails after retries.
    """
    import requests
    from typing import Dict, List

    def _get_models() -> List[Dict]:
        response = requests.get("https://api.keywordsai.co/api/models/public")
        if not response.ok:
            raise Exception(f"Failed to fetch models: {response.status_code}")

        data = response.json()
        models = [m for m in data["models"] if m.get("input_cost")]

        # Process model data
        for model in models:
            # Convert costs from per token to per million tokens
            model["input_cost"] = model["input_cost"] * 1e6
            model["output_cost"] = model["output_cost"] * 1e6

            # Normalize Google provider names
            if model["provider"]["provider_id"] in [
                "google_palm",
                "google_vertex_ai",
                "google_gemini_ai",
            ]:
                model["provider"]["provider_id"] = "google"
                model["provider"]["provider_name"] = "Google"

        return {model["model_name"]: model for model in models}

    # Use retry decorator for resilience
    retry_decorator = create_retry_decorator(
        max_retries=3, random_exponential=True, min_seconds=1, max_seconds=10
    )

    try:
        return retry_decorator(_get_models)()
    except Exception as e:
        logger.error(f"Failed to fetch models after retries: {e!s}")
        return []


def create_retry_decorator(
    max_retries: int,
    random_exponential: bool = False,
    stop_after_delay_seconds: Optional[float] = None,
    min_seconds: float = 4,
    max_seconds: float = 60,
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


def keywordsai_modelname_to_contextsize(modelname: str) -> int:
    """Calculate the maximum number of tokens possible to generate for a model."""
    models = get_keywords_models()
    return models[modelname]["max_context_window"]


def is_chat_model(model: str) -> bool:
    # TODO: check if they are chat models
    return True


def is_function_calling_model(model: str) -> bool:
    models = get_keywords_models()
    return models[model]["function_call"] == 1


def resolve_keywordsai_credentials(
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
) -> Tuple[Optional[str], str, str]:
    """ "Resolve KeywordsAI credentials.

    The order of precedence is:
    1. param
    2. env
    3. openai module
    4. default
    """
    # resolve from param or env
    api_key = get_from_param_or_env("api_key", api_key, "KEYWORDSAI_API_KEY", "")
    api_base = get_from_param_or_env("api_base", api_base, "KEYWORDSAI_API_BASE", "")
    api_version = get_from_param_or_env(
        "api_version", api_version, "KEYWORDSAI_API_VERSION", ""
    )

    # resolve from openai module or default
    final_api_key = api_key or ""
    final_api_base = api_base or DEFAULT_KEYWORDSAI_API_BASE

    return final_api_key, str(final_api_base), api_version


def validate_keywordsai_api_key(api_key: Optional[str] = None) -> None:
    openai_api_key = api_key or os.environ.get("KEYWORDSAI_API_KEY", "")

    if not openai_api_key:
        raise ValueError(MISSING_API_KEY_ERROR_MESSAGE)
