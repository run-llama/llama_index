from typing import Optional, Union
from llama_index.llms.base import LLM
from langchain.base_language import BaseLanguageModel

from llama_index.llms.langchain import LangChainLLM
from llama_index.llms.openai import OpenAI

LLMType = Union[LLM, BaseLanguageModel]


def resolve_llm(llm: Optional[LLMType] = None) -> LLM:
    if isinstance(llm, BaseLanguageModel):
        # NOTE: if it's a langchain model, wrap it in a LangChainLLM
        return LangChainLLM(llm=llm)
    
    return llm or OpenAI()
