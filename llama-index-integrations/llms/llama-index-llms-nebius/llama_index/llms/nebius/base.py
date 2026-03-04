import os
from typing import Any, Optional

from llama_index.llms.openai_like import OpenAILike


DEFAULT_API_BASE = "https://api.studio.nebius.ai/v1"


class NebiusLLM(OpenAILike):
    """
    Nebius AI Studio LLM class.

    Examples:
        `pip install llama-index-llms-nebius`

        ```python
        from llama_index.llms.nebius import NebiusLLM

        # set api key in env or in llm
        # import os
        # os.environ["NEBIUS_API_KEY"] = "your api key"

        llm = NebiusLLM(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1", api_key="your_api_key"
        )

        resp = llm.complete("Who is Paul Graham?")
        print(resp)
        ```

    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: str = DEFAULT_API_BASE,
        is_chat_model: bool = True,
        **kwargs: Any,
    ) -> None:
        api_key = api_key or os.environ.get("NEBIUS_API_KEY", None)
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
        return "NebiusLLM"
