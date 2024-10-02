import os
from typing import Any, Optional

from llama_index.llms.openai_like import OpenAILike


class Cerebras(OpenAILike):
    """
    Cerebras LLM.

    Examples:
        `pip install llama-index-llms-cerebras`

        ```python
        from llama_index.llms.cerebras import Cerebras

        # Set up the Cerebras class with the required model and API key
        llm = Cerebras(model="llama3.1-70b", api_key="your_api_key")

        # Call the complete method with a query
        response = llm.complete("Why is fast inference important?")

        print(response)
        ```
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: str = os.environ.get("CEREBRAS_BASE_URL", None)
        or "https://api.cerebras.ai/v1/",
        is_chat_model: bool = True,
        **kwargs: Any,
    ) -> None:
        api_key = api_key or os.environ.get("CEREBRAS_API_KEY", None)

        assert (
            api_key is not None
        ), "API Key not specified! Please set `CEREBRAS_API_KEY`!"

        super().__init__(
            model=model,
            api_key=api_key,
            api_base=api_base,
            is_chat_model=is_chat_model,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "Cerebras"
