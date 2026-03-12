import os
from typing import Any, Optional

from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.minimax.utils import get_context_window, FUNCTION_CALLING_MODELS

DEFAULT_API_BASE = "https://api.minimax.io/v1"
DEFAULT_MODEL = "MiniMax-M2.5"


class MiniMax(OpenAILike):
    """
    MiniMax LLM.

    MiniMax offers powerful language models with up to 204,800 tokens context window
    through an OpenAI-compatible API.

    Examples:
        `pip install llama-index-llms-minimax`

        ```python
        from llama_index.llms.minimax import MiniMax

        # Set up the MiniMax class with the required model and API key
        llm = MiniMax(model="MiniMax-M2.5", api_key="your_api_key")

        # Call the complete method with a query
        response = llm.complete("Explain the importance of low latency LLMs")

        print(response)
        ```

    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        api_base: str = DEFAULT_API_BASE,
        temperature: float = 1.0,
        **openai_llm_kwargs: Any,
    ) -> None:
        api_key = api_key or os.environ.get("MINIMAX_API_KEY", None)
        context_window = openai_llm_kwargs.pop(
            "context_window", get_context_window(model)
        )
        super().__init__(
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
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
        return "MiniMax"
