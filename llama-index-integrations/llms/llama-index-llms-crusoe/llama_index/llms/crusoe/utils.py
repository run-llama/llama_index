from __future__ import annotations

from typing import Dict

DEFAULT_CRUSOE_API_BASE = "https://managed-inference-api-proxy.crusoecloud.com/v1/"

LLAMA_MODELS: Dict[str, int] = {
    "meta-llama/Llama-3.3-70B-Instruct": 131072,
}

DEEPSEEK_MODELS: Dict[str, int] = {
    "deepseek-ai/DeepSeek-R1-0528": 163840,
    "deepseek-ai/DeepSeek-V3": 163840,
    "deepseek-ai/DeepSeek-V3-0324": 163840,
}

OPENAI_MODELS: Dict[str, int] = {
    "openai/gpt-oss-120b": 131072,
}

QWEN_MODELS: Dict[str, int] = {
    "Qwen/Qwen3-235B-A22B-Instruct-2507": 262144,
}

GOOGLE_MODELS: Dict[str, int] = {
    "google/gemma-3-12b-it": 131072,
}

MOONSHOT_MODELS: Dict[str, int] = {
    "moonshotai/Kimi-K2-Thinking": 262144,
}

ALL_AVAILABLE_MODELS: Dict[str, int] = {
    **LLAMA_MODELS,
    **DEEPSEEK_MODELS,
    **OPENAI_MODELS,
    **QWEN_MODELS,
    **GOOGLE_MODELS,
    **MOONSHOT_MODELS,
}

FUNCTION_CALLING_MODELS: frozenset[str] = frozenset(
    [
        "meta-llama/Llama-3.3-70B-Instruct",
        "openai/gpt-oss-120b",
        "deepseek-ai/DeepSeek-V3",
        "deepseek-ai/DeepSeek-V3-0324",
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "moonshotai/Kimi-K2-Thinking",
    ]
)


def crusoe_modelname_to_contextsize(modelname: str) -> int:
    """
    Calculate the maximum number of tokens possible to generate for a model.

    Args:
        modelname: The modelname we want to know the context size for.

    Returns:
        The maximum context size

    Example:
        .. code-block:: python

            max_tokens = crusoe_modelname_to_contextsize(model_name)

    """
    context_size = ALL_AVAILABLE_MODELS.get(modelname)

    if context_size is None:
        raise ValueError(
            f"Unknown model: {modelname}. Please provide a valid Crusoe model name. "
            "Known models are: " + ", ".join(ALL_AVAILABLE_MODELS.keys())
        )

    return context_size


def is_function_calling_model(model: str) -> bool:
    """Check if a model supports function calling."""
    return model in FUNCTION_CALLING_MODELS
