from typing import Any

from llama_index.bridge.pydantic import Field
from llama_index.llms import OpenAI


class OpenAILike(OpenAI):
    context_window: int = Field(
        default=512, discription="Custom context_window for custom models."
    )
    is_chat_model: bool = Field(
        default=False, description="Indicates that the custom model is a chat_model."
    )
    is_function_calling_model: bool = Field(
        default=False,
        description="Indicates that the custom model is a function calling model.",
    )

    @property
    def _is_chat_model(self) -> bool:
        return self.is_chat_model

    def _get_context_window(self) -> int:
        return self.context_window

    def __init__(self, is_chat_model=False, **data: Any) -> None:
        super().__init__(**data)

    @classmethod
    def class_name(cls) -> str:
        return "OpenAILike"
