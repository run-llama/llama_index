from typing import Any, AsyncGenerator, List
from llama_index.llms.openai_like.base import OpenAILike
from llama_index.core.base.llms.types import ChatMessage, ChatResponse


class GptOss(OpenAILike):
    """OpenAI-like LLM for gpt-oss models with thinking stream support."""

    async def astream_chat(
        self, messages: List[ChatMessage], **kwargs: Any
    ) -> AsyncGenerator[ChatResponse, None]:
        async for response in await super().astream_chat(messages, **kwargs):
            # Extract thinking stream from gpt-oss raw format
            if (
                hasattr(response.raw, "choices")
                and len(response.raw.choices) > 0
                and hasattr(response.raw.choices[0].delta, "reasoning")
            ):
                response.additional_kwargs["thinking_delta"] = response.raw.choices[
                    0
                ].delta.reasoning
            yield response
