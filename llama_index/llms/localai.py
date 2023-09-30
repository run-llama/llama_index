from typing import Any, Dict, Optional

from llama_index.bridge.pydantic import Field
from llama_index.constants import DEFAULT_CONTEXT_WINDOW
from llama_index.llms.openai import OpenAI

DEFAULT_KEY = "fake"
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8080
DEFAULT_API_BASE = f"{DEFAULT_HOST}{DEFAULT_PORT}"


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

    def __init__(
        self,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        api_key: Optional[str] = DEFAULT_KEY,
        api_base: Optional[str] = DEFAULT_API_BASE,
        **openai_kwargs: Any,
    ) -> None:
        super().__init__(api_key=api_key, api_base=api_base, **openai_kwargs)
        self.context_window = context_window  # Set in pydantic

    def _get_context_window(self) -> int:
        return self.context_window

    def _update_max_tokens(self, all_kwargs: Dict[str, Any], prompt: str) -> None:
        # This subclass only supports max_tokens via LocalAI(..., max_tokens=123)
        if self.max_tokens is not None:
            return
        all_kwargs.pop("max_tokens", None)
