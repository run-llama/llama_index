from typing import Optional, Union

from langchain.base_language import BaseLanguageModel

from llama_index.llms.base import LLM
from llama_index.llms.langchain import LangChainLLM
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.llms.openai import OpenAI

LLMType = Union[str, LLM, BaseLanguageModel]


def resolve_llm(llm: Optional[LLMType] = None) -> LLM:
    """Resolve LLM from string or LLM instance."""
    if isinstance(llm, str):
        splits = llm.split(":", 1)
        is_local = splits[0]
        model_path = splits[1] if len(splits) > 1 else None
        if is_local != "local":
            raise ValueError(
                "llm must start with str 'local' or of type LLM or BaseLanguageModel"
            )
        return LlamaCPP(
            model_path=model_path,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            model_kwargs={"n_gpu_layers": 1},
        )
    elif isinstance(llm, BaseLanguageModel):
        # NOTE: if it's a langchain model, wrap it in a LangChainLLM
        return LangChainLLM(llm=llm)

    # return default OpenAI model. If it fails, return LlamaCPP
    try:
        return llm or OpenAI()
    except ValueError:
        print(
            "******\n"
            "Could not load OpenAI model. Using default LlamaCPP=llama2-13b-chat. "
            "If you intended to use OpenAI, please check your API key."
            "\n******"
        )

        # instansiate LlamaCPP with proper llama2-chat prompts
        return LlamaCPP(
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            model_kwargs={"n_gpu_layers": 1},
        )
