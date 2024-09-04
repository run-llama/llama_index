from typing import Dict, Type

from llama_index.core.llms.llm import LLM
from llama_index.core.llms.mock import MockLLM

RECOGNIZED_LLMS: Dict[str, Type[LLM]] = {
    MockLLM.class_name(): MockLLM,
}

# Conditionals for llama-cloud support
try:
    from llama_index.llms.openai import OpenAI  # pants: no-infer-dep

    RECOGNIZED_LLMS[OpenAI.class_name()] = OpenAI  # pants: no-infer-dep
except ImportError:
    pass

try:
    from llama_index.llms.azure_openai import AzureOpenAI  # pants: no-infer-dep

    RECOGNIZED_LLMS[AzureOpenAI.class_name()] = AzureOpenAI  # pants: no-infer-dep
except ImportError:
    pass

try:
    from llama_index.llms.huggingface_api import (
        HuggingFaceInferenceAPI,
    )  # pants: no-infer-dep

    RECOGNIZED_LLMS[HuggingFaceInferenceAPI.class_name()] = HuggingFaceInferenceAPI
except ImportError:
    pass


def load_llm(data: dict) -> LLM:
    """Load LLM by name."""
    if isinstance(data, LLM):
        return data
    llm_name = data.get("class_name", None)
    if llm_name is None:
        raise ValueError("LLM loading requires a class_name")

    if llm_name not in RECOGNIZED_LLMS:
        raise ValueError(f"Invalid LLM name: {llm_name}")

    return RECOGNIZED_LLMS[llm_name].from_dict(data)
