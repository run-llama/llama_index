import os
from typing import Any, Optional

from llama_index.llms.openai_like import OpenAILike

DEFAULT_TZAFON_MODEL = "tzafon.sm-1"
DEFAULT_TZAFON_API_BASE = "https://api.tzafon.ai/v1"


class Tzafon(OpenAILike):
    """
    Tzafon LLM.

    Tzafon provides fast, reliable AI inference with an OpenAI-compatible API.

    Examples:
        `pip install llama-index-llms-tzafon`

        ```python
        from llama_index.llms.tzafon import Tzafon

        # Set up the Tzafon class with the required model and API key
        llm = Tzafon(model="tzafon.sm-1", api_key="your_api_key")

        # Or use the TZAFON_API_KEY environment variable
        llm = Tzafon(model="tzafon.sm-1")

        # Call the complete method with a query
        response = llm.complete("Explain the importance of AI safety")

        print(response)
        ```

    Available models:
        - tzafon.sm-1: Small, fast model for general tasks
        - tzafon.northstar.cua.sft: Computer-use optimized model for automation agents

    """

    def __init__(
        self,
        model: str = DEFAULT_TZAFON_MODEL,
        api_key: Optional[str] = None,
        api_base: str = DEFAULT_TZAFON_API_BASE,
        is_chat_model: bool = True,
        is_function_calling_model: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a Tzafon LLM instance.

        Args:
            model: The model to use. Defaults to "tzafon.sm-1".
            api_key: The Tzafon API key. If not provided, will look for
                TZAFON_API_KEY environment variable.
            api_base: The base URL for the Tzafon API.
                Defaults to "https://api.tzafon.ai/v1".
            is_chat_model: Whether this is a chat model. Defaults to True.
            is_function_calling_model: Whether this model supports function calling.
                Defaults to False.
            **kwargs: Additional arguments to pass to the OpenAILike base class.

        """
        api_key = api_key or os.environ.get("TZAFON_API_KEY", None)
        super().__init__(
            model=model,
            api_key=api_key,
            api_base=api_base,
            is_chat_model=is_chat_model,
            is_function_calling_model=is_function_calling_model,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "Tzafon"
