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

DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "gryphe/mythomax-l2-13b"


class OpenRouter(OpenAILike):
    """
    OpenRouter LLM.

    To instantiate the `OpenRouter` class, you will need to provide an API key. You can set the API key either as an environment variable `OPENROUTER_API_KEY` or directly in the class
    constructor. If setting it in the class constructor, it would look like this:

    If you haven't signed up for an API key yet, you can do so on the OpenRouter website at (https://openrouter.ai). Once you have your API key, you can use the `OpenRouter` class to interact
    with the LLM for tasks like chatting, streaming, and completing prompts.

    Examples:
        `pip install llama-index-llms-openrouter`

        ```python
        from llama_index.llms.openrouter import OpenRouter

        llm = OpenRouter(
            api_key="<your-api-key>",
            max_tokens=256,
            context_window=4096,
            model="gryphe/mythomax-l2-13b",
        )

        response = llm.complete("Hello World!")
        print(str(response))
        ```

    """

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
