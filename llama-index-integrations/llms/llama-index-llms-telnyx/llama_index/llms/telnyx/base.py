"""Telnyx LLM integration for LlamaIndex.

Telnyx provides an OpenAI-compatible Inference API for hosted LLMs.
This integration wraps the OpenAILike base class with Telnyx-specific defaults.
"""

import os
from typing import Any, Optional

from llama_index.llms.openai_like import OpenAILike

DEFAULT_TELNYX_API_BASE = "https://api.telnyx.com/v2/ai/openai"
DEFAULT_TELNYX_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


class Telnyx(OpenAILike):
    """Telnyx LLM.

    Telnyx provides an OpenAI-compatible Inference API for hosted LLMs
    including Llama, Qwen, DeepSeek, and more.

    To use, set the ``TELNYX_API_KEY`` environment variable or pass it
    directly via the ``api_key`` parameter.

    Examples:
        `pip install llama-index-llms-telnyx`

        ```python
        from llama_index.llms.telnyx import Telnyx

        llm = Telnyx(
            model="meta-llama/Llama-3.3-70B-Instruct",
            api_key="your_api_key",
        )

        response = llm.complete("What is Telnyx?")
        print(response)
        ```

        Streaming:
        ```python
        from llama_index.llms.telnyx import Telnyx

        llm = Telnyx(model="meta-llama/Llama-3.3-70B-Instruct")

        response = llm.stream_complete("Explain WebRTC in simple terms.")
        for chunk in response:
            print(chunk.delta, end="")
        ```

        Chat:
        ```python
        from llama_index.core.llms import ChatMessage
        from llama_index.llms.telnyx import Telnyx

        llm = Telnyx(model="meta-llama/Llama-3.3-70B-Instruct")

        messages = [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="What is Telnyx?"),
        ]
        response = llm.webchat(messages)
        print(response)
        ```
    """

    def __init__(
        self,
        model: str = DEFAULT_TELNYX_MODEL,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        api_key = api_key or os.environ.get("TELNYX_API_KEY")
        if not api_key:
            raise ValueError(
                "Telnyx API key is required. Set the TELNYX_API_KEY environment "
                "variable or pass api_key directly."
            )

        api_base = (
            api_base
            or os.environ.get("TELNYX_API_BASE")
            or DEFAULT_TELNYX_API_BASE
        )

        super().__init__(
            model=model,
            api_key=api_key,
            api_base=api_base,
            is_webchat_model=True,
            is_function_calling_model=True,
            context_window=kwargs.pop("context_window", 131072),
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "Telnyx"
