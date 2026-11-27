from dataclasses import dataclass
from typing import Any, Dict, Optional

from llama_index.core.base.llms.types import LLMMetadata
from llama_index.core.bridge.pydantic import Field
from llama_index.core.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_TEMPERATURE,
)
from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from llama_index.llms.openai_like import OpenAILike

DEFAULT_API_BASE = "https://api.sarvam.ai/v2"
DEFAULT_MODEL = "sarvam-105b"


@dataclass(frozen=True)
class SarvamModel:
    """Capability metadata for a Sarvam Token Factory chat model."""

    context_window: int
    max_output_tokens: int
    supports_tools: bool


SARVAM_MODELS: Dict[str, SarvamModel] = {
    "sarvam-105b": SarvamModel(
        context_window=128_000,
        max_output_tokens=128_000,
        supports_tools=True,
    ),
    "sarvam-105b-conversations": SarvamModel(
        context_window=32_000,
        max_output_tokens=32_000,
        supports_tools=True,
    ),
    "glm5.3": SarvamModel(
        context_window=1_048_576,
        max_output_tokens=1_048_576,
        supports_tools=True,
    ),
    "glm5.3-flash": SarvamModel(
        context_window=1_048_576,
        max_output_tokens=1_048_576,
        supports_tools=True,
    ),
    "gemma4": SarvamModel(
        context_window=262_144,
        max_output_tokens=262_144,
        supports_tools=True,
    ),
    "deepseekv4.1-flash-1017": SarvamModel(
        context_window=1_048_576,
        max_output_tokens=1_048_576,
        supports_tools=True,
    ),
    "deepseekv4.1-flash-1017-interactive": SarvamModel(
        context_window=1_048_576,
        max_output_tokens=1_048_576,
        supports_tools=True,
    ),
}


class Sarvam(OpenAILike):
    """
    Sarvam LLM.

    Sarvam Token Factory exposes an OpenAI-compatible chat completions API at
    ``https://api.sarvam.ai/v2``. Set the API key either as the environment
    variable ``SARVAM_API_KEY`` or directly in the constructor.

    Examples:
        `pip install llama-index-llms-sarvam`

        ```python
        from llama_index.llms.sarvam import Sarvam

        llm = Sarvam(
            api_key="<your-api-key>",
            model="sarvam-105b",
            reasoning_effort="none",
        )

        response = llm.complete("Hello World!")
        print(response)
        ```

    """

    model: str = Field(description="The Sarvam model to use.")
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum number of context tokens for the model.",
        gt=0,
    )
    is_chat_model: bool = Field(
        default=True,
        description=LLMMetadata.model_fields["is_chat_model"].description,
    )
    reasoning_effort: Optional[str] = Field(
        default=None,
        description=(
            "Reasoning effort for models that support it. Use 'none' to disable "
            "reasoning on Sarvam Token Factory v2."
        ),
    )

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        max_retries: int = 5,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        context_window: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        model_info = SARVAM_MODELS.get(model)
        if context_window is None:
            context_window = (
                model_info.context_window if model_info else DEFAULT_CONTEXT_WINDOW
            )

        additional_kwargs = dict(additional_kwargs or {})
        if reasoning_effort is not None:
            additional_kwargs["reasoning_effort"] = reasoning_effort

        kwargs.setdefault(
            "is_function_calling_model",
            model_info.supports_tools if model_info else False,
        )

        api_base = get_from_param_or_env(
            "api_base", api_base, "SARVAM_API_BASE", DEFAULT_API_BASE
        )
        api_key = get_from_param_or_env("api_key", api_key, "SARVAM_API_KEY")

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            context_window=context_window,
            api_base=api_base,
            api_key=api_key,
            additional_kwargs=additional_kwargs,
            max_retries=max_retries,
            reasoning_effort=reasoning_effort,
            **kwargs,
        )

    @property
    def metadata(self) -> LLMMetadata:
        model_info = SARVAM_MODELS.get(self.model)
        num_output = (
            self.max_tokens
            if self.max_tokens is not None
            else (model_info.max_output_tokens if model_info else -1)
        )
        return LLMMetadata(
            context_window=self.context_window,
            num_output=num_output,
            is_chat_model=self.is_chat_model,
            is_function_calling_model=self.is_function_calling_model,
            model_name=self.model,
        )

    @classmethod
    def class_name(cls) -> str:
        return "Sarvam_LLM"
