from typing import Any, Dict, Sequence, Tuple

from llama_index.core.base.llms.generic_utils import (
    chat_response_to_completion_response,
    stream_chat_response_to_completion_response,
    astream_chat_response_to_completion_response,
)
from llama_index.core.base.llms.types import (
    ChatMessage,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    MessageRole,
    TextBlock,
    ImageBlock,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.schema import ImageNode
from llama_index.llms.ollama import Ollama


def get_additional_kwargs(
    response: Dict[str, Any], exclude: Tuple[str, ...]
) -> Dict[str, Any]:
    return {k: v for k, v in response.items() if k not in exclude}


def _messages_to_dicts(messages: Sequence[ChatMessage]) -> Sequence[Dict[str, Any]]:
    """Convert messages to dicts.

    For use in ollama API

    """
    results = []
    for message in messages:
        # TODO: just pass through the image arg for now.
        # TODO: have a consistent interface between multimodal models
        images = message.additional_kwargs.get("images")
        results.append(
            {
                "role": message.role.value,
                "content": message.content,
                "images": images,
            }
        )
    return results


class OllamaMultiModal(Ollama):
    @classmethod
    def class_name(cls) -> str:
        return "Ollama_multi_modal_llm"

    def _get_messages(
        self, prompt: str, image_documents: Sequence[ImageNode]
    ) -> Sequence[ChatMessage]:
        image_blocks = [
            ImageBlock(
                image=image_document.image,
                path=image_document.image_path,
                url=image_document.image_url,
                image_mimetype=image_document.image_mimetype,
            )
            for image_document in image_documents
        ]

        return [
            ChatMessage(
                role=MessageRole.USER,
                blocks=[
                    TextBlock(text=prompt),
                    *image_blocks,
                ],
            )
        ]

    @llm_completion_callback()
    def complete(
        self,
        prompt: str,
        image_documents: Sequence[ImageNode],
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Complete."""
        messages = self._get_messages(prompt, image_documents)
        chat_response = self.chat(messages, **kwargs)
        return chat_response_to_completion_response(chat_response)

    @llm_completion_callback()
    def stream_complete(
        self,
        prompt: str,
        image_documents: Sequence[ImageNode],
        formatted: bool = False,
        **kwargs: Any,
    ) -> CompletionResponseGen:
        """Stream complete."""
        messages = self._get_messages(prompt, image_documents)
        stream_chat_response = self.stream_chat(messages, **kwargs)
        return stream_chat_response_to_completion_response(stream_chat_response)

    @llm_completion_callback()
    async def acomplete(
        self,
        prompt: str,
        image_documents: Sequence[ImageNode],
        **kwargs: Any,
    ) -> CompletionResponse:
        """Async complete."""
        messages = self._get_messages(prompt, image_documents)
        chat_response = await self.achat(messages, **kwargs)
        return chat_response_to_completion_response(chat_response)

    async def astream_complete(
        self,
        prompt: str,
        image_documents: Sequence[ImageNode],
        **kwargs: Any,
    ) -> CompletionResponseAsyncGen:
        """Async stream complete."""
        messages = self._get_messages(prompt, image_documents)
        astream_chat_response = await self.astream_chat(messages, **kwargs)
        return astream_chat_response_to_completion_response(astream_chat_response)
