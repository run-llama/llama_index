import os
from typing import Any, Optional

from llama_index.llms.openai_like import OpenAILike

DEFAULT_API_BASE = "https://api.hpc-ai.com/inference/v1"
DEFAULT_MODEL = "minimax/minimax-m2.5"


class HpcAiLLM(OpenAILike):
    """
    HPC-AI LLM (OpenAI-compatible API).

    HPC-AI exposes an OpenAI-compatible inference API. Common model IDs include
    ``minimax/minimax-m2.5`` and ``moonshotai/kimi-k2.5``; set ``model`` to the id
    you use on the platform.

    Examples:
        `pip install llama-index-llms-hpc-ai`

        ```python
        from llama_index.llms.hpc_ai import HpcAiLLM

        llm = HpcAiLLM(
            model="moonshotai/kimi-k2.5",
            api_key="your-api-key",
        )
        response = llm.complete("Hello!")
        print(response)
        ```

    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        is_chat_model: bool = True,
        is_function_calling_model: bool = True,
        **kwargs: Any,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("HPC_AI_API_KEY")
        if api_base is None:
            api_base = os.environ.get("HPC_AI_BASE_URL", DEFAULT_API_BASE)
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
        return "HpcAiLLM"
