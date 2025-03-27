"""ASI LLM implementation."""

import os
from typing import Any, Optional, Sequence

from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.openai.base import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    MessageRole,
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
            model (str): The ASI model to use. Defaults to "asi1-mini".
            api_key (Optional[str]): The API key to use. If None, the ASI_API_KEY
                environment variable will be used. Defaults to None.
            api_base (str): The base URL for the ASI API. Defaults to
                "https://api.asi1.ai/v1".
            is_chat_model (bool): Whether the model supports chat.
                Defaults to True.
            is_function_calling_model (bool): Whether the model supports function
                calling. Defaults to False.
            **kwargs (Any): Additional arguments to pass to the OpenAILike
                constructor.
        """
        api_key = api_key or os.environ.get("ASI_API_KEY", None)
        if api_key is None:
            raise ValueError(
                "Must specify `api_key` or set environment variable `ASI_API_KEY`."
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

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        """
        Override stream_complete to handle ASI's limitations.

        ASI doesn't support the completions endpoint at all (returns 404 error),
        so we use a fallback mechanism that returns the complete response as a
        single chunk.
        """
        # Get a complete response using the non-streaming complete method
        response = self.complete(prompt, formatted=formatted, **kwargs)

        # Create a single chunk with the complete response
        if response and response.text:
            # Create a copy of the response to avoid modifying the original
            chunk = response

            # Yield the chunk
            yield chunk

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
        # Call the parent's stream_chat method to get the raw stream
        raw_stream = super().stream_chat(messages, **kwargs)

        # Process the raw stream to extract meaningful content
        for chunk in raw_stream:
            # Check if the chunk has a delta with content
            if hasattr(chunk, "delta") and hasattr(chunk.delta, "content"):
                # If the content is None, try to extract content from other fields
                if chunk.delta.content is None:
                    # Check for 'thought' field in the raw response
                    if hasattr(chunk, "raw") and "thought" in chunk.raw:
                        # Use the 'thought' field as content
                        chunk.delta.content = chunk.raw.get("thought", "")
                    # Check for 'init_thought' field in the raw response
                    elif hasattr(chunk, "raw") and "init_thought" in chunk.raw:
                        # Use the 'init_thought' field as content
                        chunk.delta.content = chunk.raw.get("init_thought", "")

            # Yield the processed chunk
            yield chunk
