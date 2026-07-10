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

DEFAULT_API_BASE = "https://api.edenai.run/v3"
# Eden AI's EU endpoint keeps requests and data within the EU (GDPR data
# residency). Point ``api_base`` here (or set EDENAI_API_BASE) to stay in the EU.
EU_API_BASE = "https://api.eu.edenai.run/v3"
DEFAULT_MODEL = "openai/gpt-4o-mini"


class EdenAI(OpenAILike):
    """
    Eden AI LLM.

    Eden AI gives you access to 500+ models from every major provider (OpenAI,
    Anthropic, Google, Mistral, and more) behind a single, OpenAI-compatible
    API and one API key. Models are addressed in ``provider/model`` format, for
    example ``openai/gpt-4o-mini``, ``anthropic/claude-sonnet-4-5`` or
    ``mistral/mistral-large-latest``.

    Set your key with the ``EDENAI_API_KEY`` environment variable or the
    ``api_key`` argument. You can create one at https://app.edenai.run.

    For GDPR data residency, use Eden AI's EU endpoint by setting
    ``api_base="https://api.eu.edenai.run/v3"`` (or the ``EDENAI_API_BASE``
    environment variable), so requests and data stay within the EU.

    Examples:
        `pip install llama-index-llms-edenai`

        ```python
        from llama_index.llms.edenai import EdenAI

        llm = EdenAI(
            model="openai/gpt-4o-mini",
            api_key="<your-api-key>",
        )

        response = llm.complete("Hello World!")
        print(str(response))
        ```

    """

    model: str = Field(
        description=(
            "The Eden AI model to use, in 'provider/model' format. "
            "See https://app.edenai.run/models for the available options."
        )
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
    is_function_calling_model: bool = Field(
        default=True,
        description=LLMMetadata.model_fields["is_function_calling_model"].description,
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

        api_base = get_from_param_or_env(
            "api_base", api_base, "EDENAI_API_BASE", DEFAULT_API_BASE
        )
        api_key = get_from_param_or_env("api_key", api_key, "EDENAI_API_KEY", "")

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
        return "EdenAI_LLM"
