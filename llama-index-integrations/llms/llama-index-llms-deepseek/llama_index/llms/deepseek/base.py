import os
from typing import Any, Optional

from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.deepseek.utils import get_context_window, FUNCTION_CALLING_MODELS


class DeepSeek(OpenAILike):
    """
    DeepSeek LLM.

    Examples:
        `pip install llama-index-llms-deepseek`

        ```python
        from llama_index.llms.deepseek import DeepSeek

        # Set up the DeepSeek class with the required model and API key
        llm = DeepSeek(model="deepseek-chat", api_key="your_api_key")

        # Call the complete method with a query
        response = llm.complete("Explain the importance of low latency LLMs")

        print(response)
        ```
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: str = "https://api.deepseek.com",
        **openai_llm_kwargs: Any,
    ) -> None:
        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", None)
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
            **openai_llm_kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "DeepSeek"
