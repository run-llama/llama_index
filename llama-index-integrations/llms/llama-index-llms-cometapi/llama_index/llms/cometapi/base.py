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

DEFAULT_API_BASE = "https://api.cometapi.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


class CometAPI(OpenAILike):
    """
    CometAPI LLM.

    CometAPI provides access to various state-of-the-art LLM models including GPT series,
    Claude series, Gemini series, and more. To use CometAPI, you need to obtain an API key
    from https://api.cometapi.com/console/token.

    Examples:
        `pip install llama-index-llms-cometapi`

        ```python
        from llama_index.llms.cometapi import CometAPI

        llm = CometAPI(
            api_key="<your-api-key>",
            max_tokens=256,
            context_window=4096,
            model="gpt-4o-mini",
        )

        response = llm.complete("Hello World!")
        print(str(response))
        ```

    """

    model: str = Field(
        description="The CometAPI model to use. See https://api.cometapi.com/pricing for available models."
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

        api_base = get_from_param_or_env("api_base", api_base, "COMETAPI_API_BASE")
        api_key = get_from_param_or_env("api_key", api_key, "COMETAPI_API_KEY")

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
        return "CometAPI_LLM"
