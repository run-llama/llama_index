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
                if hasattr(chunk.delta, "content") and chunk.delta.content:
                    delta_content = chunk.delta.content
                elif hasattr(chunk, "raw") and chunk.raw:
                    if "thought" in chunk.raw and chunk.raw["thought"]:
                        delta_content = chunk.raw["thought"]
                    elif "init_thought" in chunk.raw and chunk.raw["init_thought"]:
                        delta_content = chunk.raw["init_thought"]
                if delta_content:
                    response = ChatResponse(
                        message=ChatMessage(
                            role=MessageRole.ASSISTANT,
                            content=(accumulated_content + delta_content),
                        ),
                        delta=delta_content,
                        raw=(chunk.raw if hasattr(chunk, "raw") else {}),
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
        raw_stream = await super(OpenAILike, self).astream_chat(messages, **kwargs)
        accumulated_content = ""
        async for chunk in raw_stream:
            delta_content = ""
            if hasattr(chunk.delta, "content") and chunk.delta.content:
                delta_content = chunk.delta.content
            elif hasattr(chunk, "raw") and chunk.raw:
                if "thought" in chunk.raw and chunk.raw["thought"]:
                    delta_content = chunk.raw["thought"]
                elif "init_thought" in chunk.raw and chunk.raw["init_thought"]:
                    delta_content = chunk.raw["init_thought"]
            if delta_content:
                response = ChatResponse(
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=(accumulated_content + delta_content),
                    ),
                    delta=delta_content,
                    raw=(chunk.raw if hasattr(chunk, "raw") else {}),
                )
                accumulated_content += delta_content
                yield response
