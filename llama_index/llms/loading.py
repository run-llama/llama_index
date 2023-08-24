from llama_index.llms.base import LLM
from llama_index.llms.custom import CustomLLM
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.langchain import LangChainLLM
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.mock import MockLLM
from llama_index.llms.openai import OpenAI
from llama_index.llms.palm import PaLM
from llama_index.llms.predibase import PredibaseLLM
from llama_index.llms.replicate import Replicate
from llama_index.llms.xinference import Xinference


def load_llm(data: dict) -> LLM:
    """Load LLM by name."""
    llm_name = data.get("class_name", None)
    if llm_name is None:
        raise ValueError("LLM loading requires a class_name")

    if llm_name == MockLLM.__name__:
        return MockLLM.from_dict(data)
    elif llm_name == Replicate.__name__:
        return Replicate.from_dict(data)
    elif llm_name == OpenAI.__name__:
        return OpenAI.from_dict(data)
    elif llm_name == HuggingFaceLLM.__name__:
        return HuggingFaceLLM.from_dict(data)
    elif llm_name == Xinference.__name__:
        return Xinference.from_dict(data)
    elif llm_name == PredibaseLLM.__name__:
        return PredibaseLLM.from_dict(data)
    elif llm_name == PaLM.__name__:
        return PaLM.from_dict(data)
    elif llm_name == LangChainLLM.__name__:
        return LangChainLLM.from_dict(data)
    elif llm_name == LlamaCPP.__name__:
        return LlamaCPP.from_dict(data)
    elif llm_name == CustomLLM.__name__:
        return CustomLLM.from_dict(data)
    else:
        raise ValueError(f"Invalid LLM name: {llm_name}")
