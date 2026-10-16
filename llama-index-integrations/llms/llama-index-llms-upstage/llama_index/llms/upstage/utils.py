import logging
from typing import Any, Dict, Optional, Tuple

from llama_index.core.base.llms.generic_utils import get_from_param_or_env

DEFAULT_UPSTAGE_API_BASE = "https://api.upstage.ai/v1/solar"
DEFAULT_CONTEXT_WINDOW = 32768
# Context lengths from https://console.upstage.ai/docs/models
# (Upstage writes "128K" for solar-pro3 as 128,000, so "512K" is 512,000).
CHAT_MODELS = {
    "solar-mini": 32768,
    "solar-pro2": 65536,
    "solar-pro3": 128000,
    "solar-pro4": 512000,
    "solar-mini4": 512000,
}

FUNCTION_CALLING_MODELS = [
    "solar-mini",
    "solar-pro2",
    "solar-pro3",
    "solar-pro4",
    "solar-mini4",
]
DOC_PARSING_MODELS = ["solar-pro2", "solar-pro3", "solar-pro4"]

ALL_AVAILABLE_MODELS = {**CHAT_MODELS}

# solar-mini4 has no public Hugging Face tokenizer yet, so it is not listed.
SOLAR_TOKENIZERS = {
    "solar-pro4": "upstage/solar-pro4-tokenizer",
    "solar-pro3": "upstage/solar-pro3-tokenizer",
    "solar-pro2": "upstage/solar-pro2-tokenizer",
    "solar-mini": "upstage/solar-1-mini-tokenizer",
}

logger = logging.getLogger(__name__)


def resolve_upstage_credentials(
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
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
