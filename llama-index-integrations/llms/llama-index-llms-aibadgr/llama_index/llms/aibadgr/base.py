import os
from typing import Any, Optional

from llama_index.llms.openai_like import OpenAILike


class AIBadgr(OpenAILike):
    """
    AI Badgr LLM (Budget/Utility, OpenAI-compatible).

    AI Badgr provides OpenAI-compatible API endpoints with tier-based model naming.
    Use tier names (basic, normal, premium) or power-user model names
    (phi-3-mini, mistral-7b, llama3-8b-instruct). OpenAI model names are also
    accepted and mapped automatically.

    Examples:
        `pip install llama-index-llms-aibadgr`

        ```python
        from llama_index.llms.aibadgr import AIBadgr

        # Set up the AIBadgr class with the required model and API key
        llm = AIBadgr(model="premium", api_key="your_api_key")

        # Call the complete method with a query
        response = llm.complete("Explain the importance of low latency LLMs")

        print(response)
        ```

    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: str = "https://aibadgr.com/api/v1",
        is_chat_model: bool = True,
        **kwargs: Any,
    ) -> None:
        api_key = api_key or os.environ.get("AIBADGR_API_KEY", None)
        api_base = os.environ.get("AIBADGR_BASE_URL", api_base)
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
        return "AIBadgr"
