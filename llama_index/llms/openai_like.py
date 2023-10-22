from types import MappingProxyType
from typing import Any, Mapping, Optional, Union

from llama_index.bridge.pydantic import Field
from llama_index.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.llms.base import LLMMetadata
from llama_index.llms.openai import OpenAI, Tokenizer


class OpenAILike(OpenAI):
    """
    OpenAILike is a thin wrapper around the OpenAI model that makes it compatible with
    3rd party tools that provide an openai-compatible api.

    Currently, llama_index prevents using custom models with their OpenAI class
    because they need to be able to infer some metadata from the model name.

    NOTE: You still need to set the OPENAI_BASE_API and OPENAI_API_KEY environment
    variables or the api_key and api_base constructor arguments.
    OPENAI_API_KEY/api_key can normally be set to anything in this case,
    but will depend on the tool you're using.
    """

    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description=LLMMetadata.__fields__["context_window"].field_info.description,
    )
    num_output: Optional[int] = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description=LLMMetadata.__fields__["num_output"].field_info.description
        + " Set to 0 or None to fall back first on max_tokens, then if max_tokens is 0"
        " or None then fall back on -1.",
    )
    is_chat_model: bool = Field(
        default=False,
        description=LLMMetadata.__fields__["is_chat_model"].field_info.description,
    )
    is_function_calling_model: bool = Field(
        default=False,
        description=LLMMetadata.__fields__[
            "is_function_calling_model"
        ].field_info.description,
    )
    tokenizer: Optional[Union[Tokenizer, str]] = Field(
        default=None,
        description=(
            "An instance of a tokenizer object that has an encode method, or the name"
            " of a tokenizer model from Hugging Face. If left as None, then this"
            " disables inference of max_tokens."
        ),
    )

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output or self.max_tokens or -1,
            is_chat_model=self.is_chat_model,
            is_function_calling_model=self.is_function_calling_model,
            model_name=self.model,
        )

    @property
    def _tokenizer(self) -> Optional[Tokenizer]:
        if isinstance(self.tokenizer, str):
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "Please install transformers (pip install transformers) to use "
                    "huggingface tokenizers with OpenAILike."
                ) from exc

            return AutoTokenizer.from_pretrained(self.tokenizer)
        return self.tokenizer

    @classmethod
    def class_name(cls) -> str:
        return "OpenAILike"


# Use these as kwargs for OpenAILike to connect to LocalAIs
localai_defaults: Mapping[str, Any] = MappingProxyType(
    {
        "api_key": "localai_fake",
        "api_type": "localai_fake",
        "api_base": "localhost:8080",
    }
)
