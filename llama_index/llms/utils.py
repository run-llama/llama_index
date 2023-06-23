from typing import Optional, Union
from llama_index.llms.base import LLM
from langchain.base_language import BaseLanguageModel

from llama_index.llms.langchain import LangChainLLM
from llama_index.llms.openai import OpenAI

LLMType = Union[LLM, BaseLanguageModel]


def resolve_llm(llm: Optional[LLMType] = None) -> LLM:
    if llm is None:
        return OpenAI()

    if isinstance(llm, BaseLanguageModel):
        return LangChainLLM(llm=llm)
    elif isinstance(llm, LLM):
        return llm
    else:
        raise ValueError(f"Invalid LLM type: {type(llm)}")
