from __future__ import annotations

from typing import Dict

DEFAULT_CRUSOE_API_BASE = "https://api.inference.crusoecloud.com/v1/"

ZAI_MODELS: Dict[str, int] = {
    "zai/GLM-5.2": 262144,
    "zai/GLM-5.1": 202000,
}

NVIDIA_MODELS: Dict[str, int] = {
    "nvidia/Nemotron-3-Nano-30B-A3B": 262144,
    "nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B": 262144,
    "nvidia/Nemotron-3-Super-120B-A12B": 262144,
    "nvidia/Nemotron-3-Ultra-550B": 262144,
}

GOOGLE_MODELS: Dict[str, int] = {
    "google/gemma-4-31b-it": 262144,
}

LLAMA_MODELS: Dict[str, int] = {
    "meta-llama/Llama-3.3-70B-Instruct": 131072,
}

DEEPSEEK_MODELS: Dict[str, int] = {
    "deepseek-ai/DeepSeek-V3-0324": 163840,
    "deepseek-ai/DeepSeek-V4-Flash": 1000000,
    "deepseek-ai/DeepSeek-V4-Pro": 1000000,
}

OPENAI_MODELS: Dict[str, int] = {
    "openai/gpt-oss-120b": 131072,
}

QWEN_MODELS: Dict[str, int] = {
    "qwen/Qwen3-235B-A22B": 131072,
}

MOONSHOT_MODELS: Dict[str, int] = {
    "moonshotai/Kimi-K2.6": 262144,
}

ALL_AVAILABLE_MODELS: Dict[str, int] = {
    **ZAI_MODELS,
    **NVIDIA_MODELS,
    **LLAMA_MODELS,
    **DEEPSEEK_MODELS,
    **OPENAI_MODELS,
    **QWEN_MODELS,
    **GOOGLE_MODELS,
    **MOONSHOT_MODELS,
}

FUNCTION_CALLING_MODELS: frozenset[str] = frozenset(
    [
        "zai/GLM-5.2",
        "zai/GLM-5.1",
        "nvidia/Nemotron-3-Nano-30B-A3B",
        "nvidia/Nemotron-3-Super-120B-A12B",
        "nvidia/Nemotron-3-Ultra-550B",
        "meta-llama/Llama-3.3-70B-Instruct",
        "openai/gpt-oss-120b",
        "deepseek-ai/DeepSeek-V3-0324",
        "deepseek-ai/DeepSeek-V4-Flash",
        "deepseek-ai/DeepSeek-V4-Pro",
        "qwen/Qwen3-235B-A22B",
        "moonshotai/Kimi-K2.6",
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
