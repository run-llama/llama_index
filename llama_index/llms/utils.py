from typing import Optional, Union

from langchain.base_language import BaseLanguageModel

from llama_index.llms.base import LLM
from llama_index.llms.langchain import LangChainLLM
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_utils import completion_to_prompt, messages_to_prompt
from llama_index.llms.mock import MockLLM
from llama_index.llms.openai import OpenAI

LLMType = Union[str, LLM, BaseLanguageModel]


def resolve_llm(llm: Optional[LLMType] = None) -> LLM:
    """Resolve LLM from string or LLM instance."""
    if llm == "default":
        # return default OpenAI model. If it fails, return LlamaCPP
        try:
            llm = OpenAI()
        except ValueError as e:
            raise ValueError(
                "******\n"
                "Could not load OpenAI model. "
                "If you intended to use OpenAI, please check your OPENAI_API_KEY.\n"
                "Original error:\n"
                f"{e!s}"
                "\nTo disable the LLM entirely, set llm=None."
                "\n******"
            )

    if isinstance(llm, str):
        splits = llm.split(":", 1)
        is_local = splits[0]
        model_path = splits[1] if len(splits) > 1 else None
        if is_local != "local":
            raise ValueError(
                "llm must start with str 'local' or of type LLM or BaseLanguageModel"
            )
        llm = LlamaCPP(
            model_path=model_path,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            model_kwargs={"n_gpu_layers": 1},
        )
    elif isinstance(llm, BaseLanguageModel):
        # NOTE: if it's a langchain model, wrap it in a LangChainLLM
        llm = LangChainLLM(llm=llm)
    elif llm is None:
        print("LLM is explicitly disabled. Using MockLLM.")
        llm = MockLLM()

    return llm
