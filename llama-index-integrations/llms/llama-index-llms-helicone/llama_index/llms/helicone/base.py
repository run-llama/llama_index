from typing import Any, Dict, Optional

from llama_index.core.bridge.pydantic import Field
from llama_index.core.constants import (
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_TEMPERATURE,
)
from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from llama_index.llms.openai_like import OpenAILike

# Default Helicone AI Gateway base. Override with HELICONE_API_BASE if needed.
DEFAULT_API_BASE = "https://ai-gateway.helicone.ai/v1"
# Default model routed via gateway; users may override to any supported provider.
DEFAULT_MODEL = "gpt-4o-mini"


class Helicone(OpenAILike):
    """
    Helicone (OpenAI-compatible) LLM.

    Route OpenAI-compatible requests through Helicone for observability and control.

    Authentication:
    - Set your Helicone API key via the `api_key` parameter or `HELICONE_API_KEY`.
      No OpenAI/third-party provider keys are required when using the AI Gateway.

    Examples:
        `pip install llama-index-llms-helicone`

        ```python
        from llama_index.llms.helicone import Helicone
        from llama_index.llms.openai_like.base import ChatMessage

        llm = Helicone(
            api_key="<helicone-api-key>",
            model="gpt-4o-mini",  # works across providers
        )

        message: ChatMessage = ChatMessage(role="user", content="Hello world!")

        response = helicone.chat(messages=[message])
        print(str(response))
        ```

    """

    model: str = Field(
        description=(
            "OpenAI-compatible model name routed via the Helicone AI Gateway. "
            "Learn more about [provider routing](https://docs.helicone.ai/gateway/provider-routing). "
            "All models are visible [here](https://www.helicone.ai/models)."
        )
    )
    api_base: Optional[str] = Field(
        default=DEFAULT_API_BASE,
        description=(
            "Base URL for the Helicone AI Gateway. Can also be set via the "
            "HELICONE_API_BASE environment variable. See the "
            "[Gateway overview](https://docs.helicone.ai/gateway/overview)."
        ),
    )
    api_key: Optional[str] = Field(
        description=(
            "Helicone API key used to authorize requests (Authorization: Bearer). "
            "Provide directly or set via HELICONE_API_KEY. Generate your API key "
            "in the [dashboard settings](https://us.helicone.ai/settings/api-keys). "
        ),
    )
    default_headers: Optional[Dict[str, str]] = Field(
        default=None,
        description=(
            "Additional HTTP headers to include with requests. The Helicone "
            "Authorization header is added automatically from api_key. See "
            "[custom properties](https://docs.helicone.ai/features/advanced-usage/custom-properties)/[headers](https://docs.helicone.ai/helicone-headers/header-directory)."
        ),
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
        default_headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        additional_kwargs = additional_kwargs or {}

        api_base = get_from_param_or_env("api_base", api_base, "HELICONE_API_BASE")
        api_key = get_from_param_or_env("api_key", api_key, "HELICONE_API_KEY")

        if default_headers:
            default_headers.update({"Authorization": f"Bearer {api_key}"})
        else:
            default_headers = {"Authorization": f"Bearer {api_key}"}

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base=api_base,
            default_headers=default_headers,
            additional_kwargs=additional_kwargs,
            max_retries=max_retries,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "Helicone_LLM"
