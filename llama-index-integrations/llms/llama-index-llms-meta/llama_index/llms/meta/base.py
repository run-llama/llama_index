# Copyright (c) Meta Platforms, Inc. and affiliates
import os
from typing import Any, Optional

from llama_index.llms.openai_like import OpenAILike


class LlamaLLM(OpenAILike):
    """
    Llama LLM.

    Examples:
        `pip install llama-index-llms-meta`

        ```python
        from llama_index.llms.meta import LlamaLLM

        # set api key in env or in llm
        # import os
        # os.environ["LLAMA_API_KEY"] = "your api key"

        llm = LlamaLLM(
            model="Llama-3.3-8B-Instruct", api_key="your_api_key"
        )

        resp = llm.complete("Who is Paul Graham?")
        print(resp)
        ```

    """

    def __init__(
        self,
        model: str = "Llama-3.3-8B-Instruct",
        api_key: Optional[str] = None,
        api_base: str = "https://api.llama.com/compat/v1",
        is_chat_model: bool = True,
        # Slightly lower to account for tokenization defaults
        context_window: int = 120000,
        **kwargs: Any,
    ) -> None:
        api_key = api_key or os.environ.get("LLAMA_API_KEY", None)
        super().__init__(
            model=model,
            api_key=api_key,
            api_base=api_base,
            is_chat_model=is_chat_model,
            context_window=context_window,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "LlamaLLM"
