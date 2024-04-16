from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from langchain.base_language import BaseLanguageModel

import os

from llama_index.core.llms.callbacks import CallbackManager
from llama_index.core.llms.llm import LLM
from llama_index.core.llms.mock import MockLLM

LLMType = Union[str, LLM, "BaseLanguageModel"]


def resolve_llm(
    llm: Optional[LLMType] = None, callback_manager: Optional[CallbackManager] = None
) -> LLM:
    """Resolve LLM from string or LLM instance."""
    from llama_index.core.settings import Settings

    try:
        from langchain.base_language import BaseLanguageModel
    except ImportError:
        BaseLanguageModel = None  # type: ignore

    if llm == "default":
        # if testing return mock llm
        if os.getenv("IS_TESTING"):
            llm = MockLLM()
            llm.callback_manager = callback_manager or Settings.callback_manager
            return llm

        # return default OpenAI model. If it fails, return LlamaCPP
        try:
            from llama_index.llms.openai import OpenAI  # pants: no-infer-dep
            from llama_index.llms.openai.utils import (
                validate_openai_api_key,
            )  # pants: no-infer-dep

            llm = OpenAI()
            validate_openai_api_key(llm.api_key)
        except ImportError:
            raise ImportError(
                "`llama-index-llms-openai` package not found, "
                "please run `pip install llama-index-llms-openai`"
            )
        except ValueError as e:
            raise ValueError(
                "\n******\n"
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
        try:
            from llama_index.llms.llama_cpp.llama_utils import (
                completion_to_prompt,
                messages_to_prompt,
            )  # pants: no-infer-dep

            from llama_index.llms.llama_cpp import LlamaCPP  # pants: no-infer-dep

            llm = LlamaCPP(
                model_path=model_path,
                messages_to_prompt=messages_to_prompt,
                completion_to_prompt=completion_to_prompt,
                model_kwargs={"n_gpu_layers": 1},
            )
        except ImportError:
            raise ImportError(
                "`llama-index-llms-llama-cpp` package not found, "
                "please run `pip install llama-index-llms-llama-cpp`"
            )

    elif BaseLanguageModel is not None and isinstance(llm, BaseLanguageModel):
        # NOTE: if it's a langchain model, wrap it in a LangChainLLM
        try:
            from llama_index.llms.langchain import LangChainLLM  # pants: no-infer-dep

            llm = LangChainLLM(llm=llm)
        except ImportError:
            raise ImportError(
                "`llama-index-llms-langchain` package not found, "
                "please run `pip install llama-index-llms-langchain`"
            )
    elif llm is None:
        print("LLM is explicitly disabled. Using MockLLM.")
        llm = MockLLM()

    llm.callback_manager = callback_manager or Settings.callback_manager

    return llm
