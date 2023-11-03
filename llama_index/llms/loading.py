from typing import Dict, Type

from llama_index.llms.base import LLM
from llama_index.llms.bedrock import Bedrock
from llama_index.llms.custom import CustomLLM
from llama_index.llms.gradient import GradientBaseModelLLM, GradientModelAdapterLLM
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.langchain import LangChainLLM
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.mock import MockLLM
from llama_index.llms.openai import OpenAI
from llama_index.llms.palm import PaLM
from llama_index.llms.predibase import PredibaseLLM
from llama_index.llms.replicate import Replicate
from llama_index.llms.vertex import Vertex
from llama_index.llms.xinference import Xinference

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
