"""ASI LLM implementation."""

import os
from typing import Any, Optional, Sequence

from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.openai.base import (
    ChatMessage,
    ChatResponseGen,
    ChatResponseAsyncGen,
    ChatResponse,
    MessageRole,
)

DEFAULT_MODEL = "asi1-mini"


class ASI(OpenAILike):
    """
    ASI LLM - Integration for ASI models.

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

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        """
        Override stream_chat to handle ASI's unique streaming format.

        ASI's streaming format includes many empty content chunks during
        the "thinking" phase before delivering the final response.

        This implementation filters out empty chunks and only yields
        chunks with actual content.
        """

        def gen() -> ChatResponseGen:
            raw_stream = super(OpenAILike, self).stream_chat(messages, **kwargs)
            accumulated_content = ""
            for chunk in raw_stream:
                delta_content = ""
                # Extract content from the chunk
                if hasattr(chunk, "raw") and chunk.raw:
                    # Check for content in choices array
                    if "choices" in chunk.raw and chunk.raw["choices"]:
                        choice = chunk.raw["choices"][0]
                        if isinstance(choice, dict):
                            if "delta" in choice and isinstance(choice["delta"], dict):
                                if (
                                    "content" in choice["delta"]
                                    and choice["delta"]["content"]
                                ):
                                    delta_content = choice["delta"]["content"]
                # Check for content in delta directly
                if not delta_content and hasattr(chunk, "delta"):
                    if hasattr(chunk.delta, "content") and chunk.delta.content:
                        delta_content = chunk.delta.content
                    elif isinstance(chunk.delta, str) and chunk.delta:
                        delta_content = chunk.delta
                if delta_content:
                    response = ChatResponse(
                        message=ChatMessage(
                            role=MessageRole.ASSISTANT,
                            content=accumulated_content + delta_content,
                        ),
                        delta=delta_content,
                        raw=chunk.raw if hasattr(chunk, "raw") else {},
                    )
                    accumulated_content += delta_content
                    yield response

        return gen()

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        """
        Override astream_chat to handle ASI's unique streaming format.

        ASI's streaming format includes many empty content chunks during
        the "thinking" phase before delivering the final response.

        This implementation filters out empty chunks and only yields
        chunks with actual content.
        """

        async def gen() -> ChatResponseAsyncGen:
            raw_stream = await super(OpenAILike, self).astream_chat(messages, **kwargs)
            accumulated_content = ""
            async for chunk in raw_stream:
                delta_content = ""
                # Extract content from the chunk
                if hasattr(chunk, "raw") and chunk.raw:
                    # Check for content in choices array
                    if "choices" in chunk.raw and chunk.raw["choices"]:
                        choice = chunk.raw["choices"][0]
                        if isinstance(choice, dict):
                            if "delta" in choice and isinstance(choice["delta"], dict):
                                if (
                                    "content" in choice["delta"]
                                    and choice["delta"]["content"]
                                ):
                                    delta_content = choice["delta"]["content"]
                # Check for content in delta directly
                if not delta_content and hasattr(chunk, "delta"):
                    if hasattr(chunk.delta, "content") and chunk.delta.content:
                        delta_content = chunk.delta.content
                    elif isinstance(chunk.delta, str) and chunk.delta:
                        delta_content = chunk.delta
                if delta_content:
                    response = ChatResponse(
                        message=ChatMessage(
                            role=MessageRole.ASSISTANT,
                            content=accumulated_content + delta_content,
                        ),
                        delta=delta_content,
                        raw=chunk.raw if hasattr(chunk, "raw") else {},
                    )
                    accumulated_content += delta_content
                    yield response

        # Return the async generator function as a coroutine to match OpenAI's pattern
        return gen()
