from typing import Optional, Tuple

from llama_index.core.base.llms.generic_utils import get_from_param_or_env

DEFAULT_FIREWORKS_API_BASE = "https://api.endpoints.fireworks.com/v1"
DEFAULT_FIREWORKS_API_VERSION = ""


def resolve_fireworks_credentials(
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
    api_key = get_from_param_or_env("api_key", api_key, "FIREWORKS_API_KEY", "")
    api_base = get_from_param_or_env("api_base", api_base, "FIREWORKS_API_BASE", "")
    api_version = get_from_param_or_env(
        "api_version", api_version, "FIREWORKS_API_VERSION", ""
    )

    # resolve from openai module or default
    final_api_key = api_key or ""
    final_api_base = api_base or DEFAULT_FIREWORKS_API_BASE
    final_api_version = api_version or DEFAULT_FIREWORKS_API_VERSION

    return final_api_key, str(final_api_base), final_api_version
