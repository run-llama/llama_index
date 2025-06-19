from typing import Any, Dict, List, Sequence

from llama_index.core.base.llms.types import (
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    MessageRole,
)
from llama_index.core.base.llms.generic_utils import (
    chat_response_to_completion_response,
    stream_chat_response_to_completion_response,
    astream_chat_response_to_completion_response,
)
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    TextBlock,
    ImageBlock,
)
from llama_index.core.schema import ImageNode
from llama_index.llms.mistralai import MistralAI


class MistralAIMultiModal(MistralAI):
    def __init__(
        self,
        model: str = "pixtral-12b-2409",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "mistral_multi_modal_llm"

    def _get_credential_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            "api_key": self.api_key,
            **kwargs,
        }

    def _get_multi_modal_chat_messages(
        self,
        prompt: str,
        role: str,
        image_documents: Sequence[ImageNode],
    ) -> List[ChatMessage]:
        blocks = []
        for image_document in image_documents:
            blocks.append(
                ImageBlock(
                    image=image_document.image,
                    path=image_document.image_path,
                    url=image_document.image_url,
                    image_mimetype=image_document.image_mimetype,
                )
            )

        blocks.append(TextBlock(text=prompt))
        return [ChatMessage(role=role, blocks=blocks)]

    def complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponse:
        messages = self._get_multi_modal_chat_messages(
            prompt=prompt, role=MessageRole.USER.value, image_documents=image_documents
        )
        chat_response = self.chat(messages=messages, **kwargs)
        return chat_response_to_completion_response(chat_response)

    def stream_complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponseGen:
        messages = self._get_multi_modal_chat_messages(
            prompt=prompt, role=MessageRole.USER.value, image_documents=image_documents
        )
        chat_response = self.stream_chat(messages=messages, **kwargs)
        return stream_chat_response_to_completion_response(chat_response)

    async def acomplete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponse:
        messages = self._get_multi_modal_chat_messages(
            prompt=prompt, role=MessageRole.USER.value, image_documents=image_documents
        )
        chat_response = await self.achat(messages=messages, **kwargs)
        return chat_response_to_completion_response(chat_response)

    async def astream_complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        messages = self._get_multi_modal_chat_messages(
            prompt=prompt, role=MessageRole.USER.value, image_documents=image_documents
        )
        chat_response = await self.astream_chat(messages=messages, **kwargs)
        return astream_chat_response_to_completion_response(chat_response)
