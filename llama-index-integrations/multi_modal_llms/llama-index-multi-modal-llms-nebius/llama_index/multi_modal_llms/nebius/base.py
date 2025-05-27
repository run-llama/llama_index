import os
from typing import Any, Optional

from llama_index.multi_modal_llms.openai import OpenAIMultiModal


DEFAULT_API_BASE = "https://api.studio.nebius.ai/v1"


class NebiusMultiModal(OpenAIMultiModal):
    """
    Nebius AI Studio Multimodal class.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: str = DEFAULT_API_BASE,
        **kwargs: Any,
    ) -> None:
        api_key = api_key or os.environ.get("NEBIUS_API_KEY", None)
        super().__init__(
            model=model,
            api_key=api_key,
            api_base=api_base,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "nebius_multi_modal_llm"

    def _get_model_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        base_kwargs = {"model": self.model, "temperature": self.temperature, **kwargs}
        if self.max_new_tokens is not None:
            base_kwargs["max_tokens"] = self.max_new_tokens
        return {**base_kwargs, **self.additional_kwargs}
