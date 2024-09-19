import logging
from typing import Any, Dict, Optional, Tuple

from llama_index.core.base.llms.generic_utils import get_from_param_or_env

DEFAULT_UPSTAGE_API_BASE = "https://api.upstage.ai/v1/solar"
DEFAULT_CONTEXT_WINDOW = 32768
CHAT_MODELS = {
    "solar-1-mini-chat": 32768,
    "solar-pro": 4096,
    "solar-docvision": 65536,
}

FUNCTION_CALLING_MODELS = ["solar-1-mini-chat"]
DOC_PARSING_MODELS = ["solar-pro"]

ALL_AVAILABLE_MODELS = {**CHAT_MODELS}

SOLAR_TOKENIZERS = {
    "solar-pro": "upstage/solar-pro-preview-tokenizer",
    "solar-1-mini-chat": "upstage/solar-1-mini-tokenizer",
    "solar-docvision": "upstage/solar-docvision-preview-tokenizer",
}

logger = logging.getLogger(__name__)


def resolve_upstage_credentials(
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """Resolve Upstage credentials.

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


def is_chat_model(model: str) -> bool:
    return True


def is_function_calling_model(model: str) -> bool:
    return model in FUNCTION_CALLING_MODELS


def is_doc_parsing_model(model: str, kwargs: Dict[str, Any]) -> bool:
    if "file_path" in kwargs:
        if model in DOC_PARSING_MODELS:
            return True
        raise ValueError("file_path is not supported for this model.")
    return False


def upstage_modelname_to_contextsize(modelname: str) -> int:
    if modelname not in ALL_AVAILABLE_MODELS:
        return DEFAULT_CONTEXT_WINDOW
    return CHAT_MODELS[modelname]
