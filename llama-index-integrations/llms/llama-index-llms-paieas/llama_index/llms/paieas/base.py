import os
from typing import Any, Optional
from urllib.parse import urljoin

from llama_index.llms.openai_like import OpenAILike

DEFAULT_MODEL_NAME = "default"


class PaiEas(OpenAILike):
    """
    PaiEas is a thin wrapper around the OpenAILike model that makes it compatible with
    Aliyun PAI-EAS(Elastic Algorithm Service) that provide effective llm services.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL_NAME,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        is_chat_model: bool = True,
        **kwargs: Any,
    ) -> None:
        api_key = api_key or os.environ.get("PAIEAS_API_KEY", None)
        api_base = api_base or os.environ.get("PAIEAS_API_BASE", None)
        super().__init__(
            model=model,
            api_key=api_key,
            api_base=urljoin(api_base, "v1"),
            is_chat_model=is_chat_model,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "PaiEasLLM"
