from typing import Optional
from llama_index.core.embeddings.utils import (
    DEFAULT_EMBED_INSTRUCTION,
    DEFAULT_QUERY_INSTRUCTION,
    DEFAULT_QUERY_BGE_INSTRUCTION_EN,
    DEFAULT_QUERY_BGE_INSTRUCTION_ZH,
    BGE_MODELS,
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
