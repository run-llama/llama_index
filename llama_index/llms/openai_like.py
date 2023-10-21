from typing import Optional

from llama_index.bridge.pydantic import Field
from llama_index.constants import DEFAULT_CONTEXT_WINDOW
from llama_index.llms.base import LLMMetadata
from llama_index.llms.openai import OpenAI, Tokenizer


class OpenAILike(OpenAI):
    """
    OpenAILike is a thin wrapper around the OpenAI model that makes it compatible with \
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
        description="The maximum number of context tokens for the model.",
    )
    is_chat_model: bool = Field(
        default=False, description="Indicates that the custom model is a chat_model."
    )
    is_function_calling_model: bool = Field(
        default=False,
        description="Indicates that the custom model is a function calling model.",
    )
    tokenizer: Optional[Tokenizer] = Field(
        default=None,
        description="An instance of a tokenizer object that has an encode method. "
        "If not provided, will default to the huggingface tokenizer for the model.",
    )

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens or -1,
            is_chat_model=self.is_chat_model,
            is_function_calling_model=self.is_function_calling_model,
            model_name=self.model,
        )

    @property
    def _tokenizer(self) -> Tokenizer:
        if not self.tokenizer:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "Please install transformers (pip install transformers) to use "
                    "huggingface tokenizers with OpenAILike."
                ) from exc

            return AutoTokenizer.from_pretrained(self._get_model_name())
        else:
            return self.tokenizer

    @classmethod
    def class_name(cls) -> str:
        return "OpenAILike"
