from typing import Any, Dict, Optional

from llama_index.legacy.bridge.pydantic import Field
from llama_index.legacy.constants import (
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_TEMPERATURE,
)
from llama_index.legacy.core.llms.types import LLMMetadata
from llama_index.legacy.llms.generic_utils import get_from_param_or_env
from llama_index.legacy.llms.openai_like import OpenAILike

DEFAULT_API_BASE = "https://router.neutrinoapp.com/api/llm-router"
DEFAULT_ROUTER = "default"
MAX_CONTEXT_WINDOW = 200000


class Neutrino(OpenAILike):
    model: str = Field(
        description="The Neutrino router to use. See https://docs.neutrinoapp.com/router for details."
    )
    context_window: int = Field(
        default=MAX_CONTEXT_WINDOW,
        description="The maximum number of context tokens for the model. Defaults to the largest supported model (Claude).",
        gt=0,
    )
    is_chat_model: bool = Field(
        default=True,
        description=LLMMetadata.__fields__["is_chat_model"].field_info.description,
    )

    def __init__(
        self,
        model: Optional[str] = None,
        router: str = DEFAULT_ROUTER,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_NUM_OUTPUTS,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        max_retries: int = 5,
        api_base: Optional[str] = DEFAULT_API_BASE,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        additional_kwargs = additional_kwargs or {}

        api_base = get_from_param_or_env("api_base", api_base, "NEUTRINO_API_BASE")
        api_key = get_from_param_or_env("api_key", api_key, "NEUTRINO_API_KEY")

        model = model or router

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
        return "Neutrino_LLM"
