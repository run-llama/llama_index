from typing import Optional

import requests

from llama_index.core.embeddings.utils import (
    BGE_MODELS,
    DEFAULT_EMBED_INSTRUCTION,
    DEFAULT_QUERY_BGE_INSTRUCTION_EN,
    DEFAULT_QUERY_BGE_INSTRUCTION_ZH,
    DEFAULT_QUERY_INSTRUCTION,
    INSTRUCTOR_MODELS,
)


def get_query_instruct_for_model_name(model_name: Optional[str]) -> str:
    """Get query text instruction for a given model name."""
    if model_name in INSTRUCTOR_MODELS:
        return DEFAULT_QUERY_INSTRUCTION
    if model_name in BGE_MODELS:
        if "zh" in model_name:
            return DEFAULT_QUERY_BGE_INSTRUCTION_ZH
        return DEFAULT_QUERY_BGE_INSTRUCTION_EN
    return ""


def format_query(
    query: str, model_name: Optional[str], instruction: Optional[str] = None
) -> str:
    if instruction is None:
        instruction = get_query_instruct_for_model_name(model_name)
    # NOTE: strip() enables backdoor for defeating instruction prepend by
    # passing empty string
    return f"{instruction} {query}".strip()


def get_text_instruct_for_model_name(model_name: Optional[str]) -> str:
    """Get text instruction for a given model name."""
    return DEFAULT_EMBED_INSTRUCTION if model_name in INSTRUCTOR_MODELS else ""


def format_text(
    text: str, model_name: Optional[str], instruction: Optional[str] = None
) -> str:
    if instruction is None:
        instruction = get_text_instruct_for_model_name(model_name)
    # NOTE: strip() enables backdoor for defeating instruction prepend by
    # passing empty string
    return f"{instruction} {text}".strip()


def get_pooling_mode(model_name: Optional[str]) -> str:
    pooling_config_url = (
        f"https://huggingface.co/{model_name}/raw/main/1_Pooling/config.json"
    )

    try:
        response = requests.get(pooling_config_url)
        config_data = response.json()

        cls_token = config_data.get("pooling_mode_cls_token", False)
        mean_tokens = config_data.get("pooling_mode_mean_tokens", False)

        if mean_tokens:
            return "mean"
        elif cls_token:
            return "cls"
    except requests.exceptions.RequestException:
        print(
            "Warning: Pooling config file not found; pooling mode is defaulted to 'cls'."
        )
    return "cls"
