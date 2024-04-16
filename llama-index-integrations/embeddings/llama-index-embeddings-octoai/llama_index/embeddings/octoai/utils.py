from typing import Optional, Tuple

from llama_index.core.base.llms.generic_utils import get_from_param_or_env

DEFAULT_OCTOAI_API_BASE = "https://text.octoai.run/v1"
DEFAULT_OCTOAI_API_VERSION = ""
DEFAULT_OCTOAI_EMBED_MODEL = "thenlper/gte-large"
DEFAULT_OCTOAI_EMBED_BATCH_SIZE = 2048


def resolve_octoai_credentials(
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
) -> Tuple[Optional[str], str, str]:
    """
    "Resolve OctoAI credentials.

    The order of precedence is:
    1. param
    2. env
    4. octoai default
    """
    # resolve from param or env
    api_key = get_from_param_or_env("api_key", api_key, "OCTOAI_API_KEY", "")
    api_base = get_from_param_or_env(
        "api_base", api_base, "OCTOAI_API_BASE", DEFAULT_OCTOAI_API_BASE
    )
    api_version = get_from_param_or_env(
        "api_version", api_version, "OCTOAI_API_VERSION", DEFAULT_OCTOAI_API_VERSION
    )

    return api_key, str(api_base), api_version
