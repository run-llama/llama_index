import os
from typing import Any, Optional

from llama_index.llms.openai_like import OpenAILike


class Groq(OpenAILike):
    """
    Groq LLM.

    Examples:
        `pip install llama-index-llms-groq`

        ```python
        from llama_index.llms.groq import Groq

        # Set up the Groq class with the required model and API key
        llm = Groq(model="llama3-70b-8192", api_key="your_api_key")

        # Call the complete method with a query
        response = llm.complete("Explain the importance of low latency LLMs")

        print(response)
        ```
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: str = "https://api.groq.com/openai/v1",
        is_chat_model: bool = True,
        is_function_calling_model: bool = True,
        **kwargs: Any,
    ) -> None:
        api_key = api_key or os.environ.get("GROQ_API_KEY", None)
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
        return "Groq"
