from typing import Any, Dict, Optional

from llama_index.core.base.llms.types import LLMMetadata
from llama_index.core.bridge.pydantic import Field
from llama_index.core.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_TEMPERATURE,
)
from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from llama_index.llms.openai_like import OpenAILike

DEFAULT_API_BASE = "https://api.apertis.ai/v1"
DEFAULT_MODEL = "gpt-5.2"


class Apertis(OpenAILike):
    """
    Apertis LLM.

    Apertis provides a unified API gateway to access multiple LLM providers
    including OpenAI, Anthropic, Google, and more through an OpenAI-compatible
    interface.

    Supported Endpoints:
        - `/v1/chat/completions` - OpenAI Chat Completions format (default)
        - `/v1/responses` - OpenAI Responses format compatible
        - `/v1/messages` - Anthropic format compatible

    To instantiate the `Apertis` class, you will need an API key. You can set
    the API key either as an environment variable `APERTIS_API_KEY` or directly
    in the class constructor.

    You can obtain an API key at https://api.apertis.ai/token

    Examples:
        `pip install llama-index-llms-apertis`

        ```python
        from llama_index.llms.apertis import Apertis

        llm = Apertis(
            api_key="<your-api-key>",
            model="gpt-5.2",
        )

        response = llm.complete("Hello World!")
        print(str(response))
        ```

    """

    model: str = Field(
        description="The model to use. Supports models from OpenAI, Anthropic, Google, and more."
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum number of context tokens for the model.",
        gt=0,
    )
    is_chat_model: bool = Field(
        default=True,
        description=LLMMetadata.model_fields["is_chat_model"].description,
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

        api_base = get_from_param_or_env("api_base", api_base, "APERTIS_API_BASE")
        api_key = get_from_param_or_env("api_key", api_key, "APERTIS_API_KEY")

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
        return "Apertis_LLM"
