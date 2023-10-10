from typing import Any, Dict, Optional

from llama_index.bridge.pydantic import Field
from llama_index.constants import DEFAULT_CONTEXT_WINDOW
from llama_index.llms.openai import OpenAI

DEFAULT_API_KEY = "fake"
DEFAULT_API_HOST = "localhost"
DEFAULT_API_PORT = 8080
DEFAULT_API_BASE = f"{DEFAULT_API_HOST}{DEFAULT_API_PORT}"


class LocalAI(OpenAI):
    """
    LocalAI is a free, open source, and self-hosted OpenAI alternative.

    Docs: https://localai.io/
    Source: https://github.com/go-skynet/LocalAI
    """

    @classmethod
    def class_name(cls) -> str:
        return "LocalAI"

    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum number of context tokens for the model.",
    )
    globally_use_chat_completions: Optional[bool] = Field(
        default=None,
        description=(
            "Set None (default) to per-invocation decide on using /chat/completions"
            " vs /completions endpoints with query keyword arguments,"
            " set False to universally use /completions endpoint,"
            " set True to universally use /chat/completions endpoint."
        ),
    )

    def __init__(
        self,
        api_key: Optional[str] = DEFAULT_API_KEY,
        api_base: Optional[str] = DEFAULT_API_BASE,
        **kwargs: Any,
    ) -> None:
        super().__init__(api_key=api_key, api_base=api_base, **kwargs)

    def _get_context_window(self) -> int:
        return self.context_window

    def _update_max_tokens(self, all_kwargs: Dict[str, Any], prompt: str) -> None:
        # This subclass only supports max_tokens via LocalAI(..., max_tokens=123)
        if self.max_tokens is not None:
            return

    @property
    def _is_chat_model(self) -> bool:
        if self.globally_use_chat_completions is not None:
            return self.globally_use_chat_completions
        raise NotImplementedError(
            "Inferring of when to use /chat/completions is unsupported by"
            f" {type(self).__name__}. Please either set 'globally_use_chat_completions'"
            " arg during construction, or pass the arg 'use_chat_completions' in your"
            " query, setting True for /chat/completions or False for /completions."
        )
