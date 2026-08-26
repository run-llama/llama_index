import os
from typing import Any, Optional

from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.volcengine.utils import (
    FUNCTION_CALLING_MODELS,
    get_context_window,
)

DEFAULT_API_BASE = "https://ark.cn-beijing.volces.com/api/v3"


class Volcengine(OpenAILike):
    """
    Volcengine Ark LLM.

    Examples:
        `pip install llama-index-llms-volcengine`

        ```python
        from llama_index.llms.volcengine import Volcengine

        # Set up the Volcengine class with the required model and API key
        llm = Volcengine(
            model="doubao-seed-2-1-pro-260628", api_key="your_api_key"
        )

        # Call the complete method with a query
        response = llm.complete("Explain the importance of low latency LLMs")

        print(response)
        ```

    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: str = DEFAULT_API_BASE,
        **openai_llm_kwargs: Any,
    ) -> None:
        api_key = api_key or os.environ.get("ARK_API_KEY", None)
        context_window = openai_llm_kwargs.pop(
            "context_window", get_context_window(model)
        )
        super().__init__(
            model=model,
            api_key=api_key,
            api_base=api_base,
            is_chat_model=openai_llm_kwargs.pop("is_chat_model", True),
            is_function_calling_model=openai_llm_kwargs.pop(
                "is_function_calling_model", model in FUNCTION_CALLING_MODELS
            ),
            context_window=context_window,
            **openai_llm_kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "Volcengine"
