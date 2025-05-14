from typing import Optional, Tuple

from llama_index.core.base.llms.generic_utils import get_from_param_or_env

DEFAULT_API_BASE = "https://api.studio.nebius.ai/v1"
DEFAULT_NEBIUS_API_VERSION = ""


def resolve_nebius_credentials(
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
) -> Tuple[Optional[str], str, str]:
    """
    "Resolve Nebius AI Studio credentials.

    The order of precedence is:
    1. param
    2. env
    3. openai module
    4. default
    """
    # resolve from param or env
    api_key = get_from_param_or_env("api_key", api_key, "NEBIUS_API_KEY", "")
    api_base = get_from_param_or_env(
        "api_base", api_base, "NEBIUS_API_BASE", DEFAULT_API_BASE
    )
    api_version = get_from_param_or_env(
        "api_version", api_version, "NEBIUS_API_VERSION", DEFAULT_NEBIUS_API_VERSION
    )
    return api_key, str(api_base), api_version
