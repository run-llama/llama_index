from typing import Any, Dict, Optional

from llama_index.legacy.bridge.pydantic import Field
from llama_index.legacy.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_TEMPERATURE,
)
from llama_index.legacy.core.llms.types import LLMMetadata
from llama_index.legacy.llms.generic_utils import get_from_param_or_env
from llama_index.legacy.llms.openai_like import OpenAILike

DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "gryphe/mythomax-l2-13b"


class OpenRouter(OpenAILike):
    model: str = Field(
        description="The OpenRouter model to use. See https://openrouter.ai/models for options."
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum number of context tokens for the model. See https://openrouter.ai/models for options.",
        gt=0,
    )
    is_chat_model: bool = Field(
        default=True,
        description=LLMMetadata.__fields__["is_chat_model"].field_info.description,
    )

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_NUM_OUTPUTS,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        max_retries: int = 5,
        api_base: Optional[str] = DEFAULT_API_BASE,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        additional_kwargs = additional_kwargs or {}

        api_base = get_from_param_or_env("api_base", api_base, "OPENROUTER_API_BASE")
        api_key = get_from_param_or_env("api_key", api_key, "OPENROUTER_API_KEY")

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base=api_base,
            api_key=api_key,
            additional_kwargs=additional_kwargs,
            max_retries=max_retries,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "OpenRouter_LLM"
