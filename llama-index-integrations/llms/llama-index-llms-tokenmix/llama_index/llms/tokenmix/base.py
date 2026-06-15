from typing import Any, Dict, Optional

from llama_index.core.base.llms.generic_utils import (
    get_from_param_or_env,
)
from llama_index.llms.openai_like import OpenAILike

DEFAULT_API_BASE = "https://api.tokenmix.ai/v1"
DEFAULT_MODEL = "deepseek/deepseek-v4-pro"


def is_function_calling_model(model: str) -> bool:
    """Return True for TokenMix model families that support tool / function calling."""
    function_calling_families = {
        "deepseek",
        "qwen",
        "kimi",
        "moonshot",
        "glm",
        "minimax",
    }
    return any(family in model for family in function_calling_families)


class TokenMix(OpenAILike):
    """
    TokenMix LLM.

    TokenMix is an OpenAI-compatible API gateway that exposes DeepSeek, Qwen, Kimi,
    GLM, MiniMax and other models through a single endpoint
    (``https://api.tokenmix.ai/v1``) and one API key.

    Because the gateway is OpenAI-compatible, this integration is a thin wrapper over
    :class:`~llama_index.llms.openai_like.OpenAILike`. Set ``model`` to the full
    TokenMix model name (for example ``deepseek/deepseek-v4-pro``) and provide your
    API key via the ``api_key`` argument or the ``TOKENMIX_API_KEY`` environment
    variable.

    Examples:
        ``pip install llama-index-llms-tokenmix``

        ```python
        from llama_index.llms.tokenmix import TokenMix

        llm = TokenMix(model="deepseek/deepseek-v4-pro", api_key="YOUR_KEY")
        response = llm.complete("Hello")
        print(response)
        ```

    Get an API key and browse the model catalog at https://tokenmix.ai/models.

    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        is_chat_model: bool = True,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        additional_kwargs = additional_kwargs or {}
        api_base = get_from_param_or_env(
            "api_base", api_base, "TOKENMIX_API_BASE", DEFAULT_API_BASE
        )
        api_key = get_from_param_or_env("api_key", api_key, "TOKENMIX_API_KEY")

        super().__init__(
            api_base=api_base,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            is_chat_model=is_chat_model,
            is_function_calling_model=is_function_calling_model(model),
            additional_kwargs=additional_kwargs,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "TokenMix"
