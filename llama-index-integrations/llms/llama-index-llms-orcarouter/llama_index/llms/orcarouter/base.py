from typing import Any, Dict, List, Optional

from llama_index.core.base.llms.types import LLMMetadata
from llama_index.core.bridge.pydantic import Field
from llama_index.core.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_TEMPERATURE,
)
from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from llama_index.llms.openai_like import OpenAILike

DEFAULT_API_BASE = "https://api.orcarouter.ai/v1"
DEFAULT_MODEL = "orcarouter/auto"

ATTRIBUTION_REFERER = "https://www.llamaindex.ai/"
ATTRIBUTION_TITLE = "LlamaIndex"


class OrcaRouter(OpenAILike):
    """
    OrcaRouter LLM.

    OrcaRouter is an OpenAI-compatible meta-router that exposes 150+ upstream models
    behind a single API key, with an adaptive router (`orcarouter/auto`) that picks
    the cheapest / fastest / highest-quality upstream per request based on a learned
    contextual bandit policy.

    To instantiate the `OrcaRouter` class, you will need to provide an API key.
    Set the environment variable `ORCAROUTER_API_KEY` or pass `api_key` directly.
    Register and obtain a key at https://www.orcarouter.ai.

    Reference: https://docs.orcarouter.ai
    Full model catalog: https://www.orcarouter.ai/models

    Examples:
        `pip install llama-index-llms-orcarouter`

        ```python
        from llama_index.llms.orcarouter import OrcaRouter

        # Default: the adaptive `orcarouter/auto` router.
        llm = OrcaRouter(
            api_key="<your-api-key>",
            max_tokens=256,
            context_window=128000,
        )

        response = llm.complete("Hello World!")
        print(str(response))
        ```

        Pin a specific upstream model:

        ```python
        llm = OrcaRouter(
            api_key="<your-api-key>",
            model="anthropic/claude-opus-4.7",
        )
        ```

        Fallback list (`extra_body` routing preference):

        ```python
        llm = OrcaRouter(
            api_key="<your-api-key>",
            model="openai/gpt-4o-mini",
            fallback_models=["openai/gpt-4o", "anthropic/claude-sonnet-4.6"],
        )
        ```

    """

    model: str = Field(
        description=(
            "The OrcaRouter model to use, e.g. `orcarouter/auto` for the adaptive "
            "router, or a pinned upstream like `openai/gpt-4o`. "
            "See https://www.orcarouter.ai/models for the full catalog."
        )
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description=(
            "The maximum number of context tokens for the model. "
            "See https://www.orcarouter.ai/models for per-model limits."
        ),
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
        fallback_models: Optional[List[str]] = None,
        default_headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        additional_kwargs = additional_kwargs or {}

        if fallback_models is not None:
            extra_body = additional_kwargs.get("extra_body")
            if extra_body is None:
                extra_body = {}
                additional_kwargs["extra_body"] = extra_body
            extra_body.setdefault("models", list(fallback_models))
            extra_body.setdefault("route", "fallback")

        # Attribution headers so OrcaRouter can attribute traffic to LlamaIndex.
        # User-supplied headers take precedence.
        headers = {
            "HTTP-Referer": ATTRIBUTION_REFERER,
            "X-Title": ATTRIBUTION_TITLE,
        }
        if default_headers:
            headers.update(default_headers)

        api_base = get_from_param_or_env(
            "api_base", api_base, "ORCAROUTER_API_BASE_URL", DEFAULT_API_BASE
        )
        api_key = get_from_param_or_env("api_key", api_key, "ORCAROUTER_API_KEY")

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base=api_base,
            api_key=api_key,
            additional_kwargs=additional_kwargs,
            max_retries=max_retries,
            default_headers=headers,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "OrcaRouter_LLM"
