from typing import Dict, Type

from llama_index.legacy.llms.bedrock import Bedrock
from llama_index.legacy.llms.custom import CustomLLM
from llama_index.legacy.llms.gradient import (
    GradientBaseModelLLM,
    GradientModelAdapterLLM,
)
from llama_index.legacy.llms.huggingface import HuggingFaceLLM
from llama_index.legacy.llms.langchain import LangChainLLM
from llama_index.legacy.llms.llama_cpp import LlamaCPP
from llama_index.legacy.llms.llm import LLM
from llama_index.legacy.llms.mock import MockLLM
from llama_index.legacy.llms.openai import OpenAI
from llama_index.legacy.llms.palm import PaLM
from llama_index.legacy.llms.predibase import PredibaseLLM
from llama_index.legacy.llms.replicate import Replicate
from llama_index.legacy.llms.vertex import Vertex
from llama_index.legacy.llms.xinference import Xinference

RECOGNIZED_LLMS: Dict[str, Type[LLM]] = {
    MockLLM.class_name(): MockLLM,
    Replicate.class_name(): Replicate,
    HuggingFaceLLM.class_name(): HuggingFaceLLM,
    OpenAI.class_name(): OpenAI,
    Xinference.class_name(): Xinference,
    LlamaCPP.class_name(): LlamaCPP,
    LangChainLLM.class_name(): LangChainLLM,
    PaLM.class_name(): PaLM,
    PredibaseLLM.class_name(): PredibaseLLM,
    Bedrock.class_name(): Bedrock,
    CustomLLM.class_name(): CustomLLM,
    GradientBaseModelLLM.class_name(): GradientBaseModelLLM,
    GradientModelAdapterLLM.class_name(): GradientModelAdapterLLM,
    Vertex.class_name(): Vertex,
}


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
