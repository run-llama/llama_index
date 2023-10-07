from typing import Any, Optional

from llama_index.bridge.pydantic import Field
from llama_index.constants import DEFAULT_CONTEXT_WINDOW
from llama_index.llms import OpenAI


class OpenAILike(OpenAI):
    """
    OpenAILike is a thin wrapper around the OpenAI model that makes it compatible with 3rd party tools
    that provide an openai-compatible api

    Currently, llama_index prevents using custom models with their OpenAI class because they need to be able to infer some metadata from the model name.

    NOTE: You still need to set the OPENAI_BASE_API and OPENAI_KEY environment variables. OPENAI_KEY can normally be set to anything in this case, but will depend on the tool you're using.
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
    tokenizer: Optional[str] = Field(
        default=None,
        description="The name of a huggingface transformers tokenizer to use for this model",
    )

    def __init__(self, is_chat_model: bool = False, **data: Any) -> None:
        super().__init__(**data)

    @property
    def _is_chat_model(self) -> bool:
        return self.is_chat_model

    def _get_context_window(self) -> int:
        return self.context_window

    def _get_encoding_for_model(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self._get_model_name())

    def _get_is_function_calling_model(self):
        return self.is_function_calling_model

    @classmethod
    def class_name(cls) -> str:
        return "OpenAILike"
