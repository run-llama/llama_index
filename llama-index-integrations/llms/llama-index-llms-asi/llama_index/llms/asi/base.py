"""ASI LLM implementation."""

import os
from typing import Any, Optional, Sequence

from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.openai.base import (
    ChatMessage,
    ChatResponseGen,
    ChatResponseAsyncGen,
)

DEFAULT_MODEL = "asi1-mini"


class ASI(OpenAILike):
    """ASI LLM - Integration for ASI models.

    Currently supported models:
    - asi1-mini

    Examples:
        `pip install llama-index-llms-asi`

        ```python
        from llama_index.llms.asi import ASI

        # Set up the ASI class with the required model and API key
        llm = ASI(model="asi1-mini", api_key="your_api_key")

        # Call the complete method with a query
        response = llm.complete("Explain the importance of AI")

        print(response)
        ```
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        api_base: str = "https://api.asi1.ai/v1",
        is_chat_model: bool = True,
        is_function_calling_model: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the ASI LLM.

        Args:
            model (str): The ASI model to use.
            api_key (Optional[str]): The API key to use.
            api_base (str): The base URL for the ASI API.
            is_chat_model (bool): Whether the model supports chat.
            is_function_calling_model (bool): Whether the model supports
                function calling.
            **kwargs (Any): Additional arguments to pass to the OpenAILike
                constructor.
        """
        api_key = api_key or os.environ.get("ASI_API_KEY", None)
        if api_key is None:
            raise ValueError(
                "Must specify `api_key` or set environment variable " "`ASI_API_KEY`."
            )

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
        return "ASI"

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        """
        Override stream_chat to handle ASI's unique streaming format.

        ASI's streaming format includes custom fields like 'thought' and
        'init_thought' that aren't part of the standard OpenAI format.
        This method processes the raw stream to extract meaningful content
        from these fields if available.
        """

        def gen() -> ChatResponseGen:
            # Call the parent's stream_chat method to get the raw stream
            raw_stream = super(OpenAILike, self).stream_chat(messages, **kwargs)

            # Process the raw stream to extract meaningful content
            for chunk in raw_stream:
                if hasattr(chunk, "delta") and hasattr(chunk.delta, "content"):
                    if chunk.delta.content is None:
                        if hasattr(chunk, "raw") and "thought" in chunk.raw:
                            chunk.delta.content = chunk.raw.get("thought", "")
                        elif hasattr(chunk, "raw") and "init_thought" in chunk.raw:
                            chunk.delta.content = chunk.raw.get("init_thought", "")

                yield chunk

        return gen()

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        """
        Override astream_chat to handle ASI's unique streaming format (async).

        ASI's streaming format includes custom fields like 'thought' and
        'init_thought' that aren't part of the standard OpenAI format.
        This method processes the raw stream to extract meaningful content
        from these fields if available.
        """

        async def gen() -> ChatResponseAsyncGen:
            # Call the parent's astream_chat method to get the raw stream
            raw_stream = await super(OpenAILike, self).astream_chat(messages, **kwargs)

            # Process the raw stream to extract meaningful content
            async for chunk in raw_stream:
                if hasattr(chunk, "delta") and hasattr(chunk.delta, "content"):
                    if chunk.delta.content is None:
                        if hasattr(chunk, "raw") and "thought" in chunk.raw:
                            chunk.delta.content = chunk.raw.get("thought", "")
                        elif hasattr(chunk, "raw") and "init_thought" in chunk.raw:
                            chunk.delta.content = chunk.raw.get("init_thought", "")

                yield chunk

        return gen()
