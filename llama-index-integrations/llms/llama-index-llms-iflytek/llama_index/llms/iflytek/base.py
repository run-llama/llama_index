from typing import Any, Dict, Optional

from llama_index.core.base.llms.generic_utils import (
    get_from_param_or_env,
)
from llama_index.llms.openai_like import OpenAILike

DEFAULT_API_BASE = "https://spark-api-open.xf-yun.com/v1"
DEFAULT_MODEL = "generalv3.5"


def is_function_calling_model(model: str) -> bool:
    function_calling_models = {
        "4.0Ultra",
        "generalv3.5",
        "max-32k",
    }
    return any(model_name in model for model_name in function_calling_models)


class IFlytek(OpenAILike):
    """
    iFlytek Spark LLM.

    iFlytek Spark exposes an OpenAI-compatible chat-completions API
    (https://spark-api-open.xf-yun.com/v1), so this integration builds on
    ``OpenAILike``. Provide the HTTP API password from the iFlytek open platform
    console as ``api_key`` (or via the ``IFLYTEK_API_KEY`` environment variable)
    and pick a model such as ``generalv3.5``, ``4.0Ultra`` or ``lite``.

    Examples:
        `pip install llama-index-llms-iflytek`

        ```python
        from llama_index.llms.iflytek import IFlytek

        llm = IFlytek(model="4.0Ultra", api_key="your-api-password")
        response = llm.complete("你好")
        print(response)
        ```

    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        is_chat_model: bool = True,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        additional_kwargs = additional_kwargs or {}
        api_base = get_from_param_or_env(
            "api_base", DEFAULT_API_BASE, "IFLYTEK_API_BASE"
        )
        api_key = get_from_param_or_env("api_key", api_key, "IFLYTEK_API_KEY")

        super().__init__(
            api_base=api_base,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            is_chat_model=is_chat_model,
            is_function_calling_model=is_function_calling_model(model),
            additional_kwargs=additional_kwargs,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "IFlytek"
