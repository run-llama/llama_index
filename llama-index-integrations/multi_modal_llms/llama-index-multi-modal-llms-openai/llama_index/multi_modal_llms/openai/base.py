from typing import Any, Optional, Sequence
from pathlib import Path

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
    ImageBlock,
)
from llama_index.core.schema import ImageNode
from llama_index.llms.openai import OpenAI


class OpenAIMultiModal(OpenAI):
    @classmethod
    def class_name(cls) -> str:
        return "openai_multi_modal_llm"

    def _get_multi_modal_chat_message(
        self,
        prompt: str,
        role: str,
        image_documents: Sequence[ImageNode],
        image_detail: Optional[str] = "low",
        **kwargs: Any,
    ) -> ChatMessage:
        chat_msg = ChatMessage(role=role, content=prompt)
        if not image_documents:
            # if image_documents is empty, return text only chat message
            return chat_msg

        for image_document in image_documents:
            # Create the appropriate ContentBlock depending on the document content
            if image_document.image:
                chat_msg.blocks.append(
                    ImageBlock(
                        image=bytes(image_document.image, encoding="utf-8"),
                        detail=image_detail,
                    )
                )
            elif image_document.image_url:
                chat_msg.blocks.append(
                    ImageBlock(url=image_document.image_url, detail=image_detail)
                )
            elif image_document.image_path:
                chat_msg.blocks.append(
                    ImageBlock(
                        path=Path(image_document.image_path),
                        detail=image_detail,
                        image_mimetype=image_document.image_mimetype
                        or image_document.metadata.get("file_type"),
                    )
                )
            elif f_path := image_document.metadata.get("file_path"):
                chat_msg.blocks.append(
                    ImageBlock(
                        path=Path(f_path),
                        detail=image_detail,
                        image_mimetype=image_document.metadata.get("file_type"),
                    )
                )

        return chat_msg

    def complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponse:
        chat_message = self._get_multi_modal_chat_message(
            prompt=prompt,
            role=MessageRole.USER,
            image_documents=image_documents,
        )
        chat_response = self.chat([chat_message], **kwargs)
        return chat_response_to_completion_response(chat_response)

    def stream_complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponseGen:
        chat_message = self._get_multi_modal_chat_message(
            prompt=prompt,
            role=MessageRole.USER,
            image_documents=image_documents,
        )
        chat_response = self.stream_chat([chat_message], **kwargs)
        return stream_chat_response_to_completion_response(chat_response)

    # ===== Async Endpoints =====

    async def acomplete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponse:
        chat_message = self._get_multi_modal_chat_message(
            prompt=prompt,
            role=MessageRole.USER,
            image_documents=image_documents,
        )
        chat_response = await self.achat([chat_message], **kwargs)
        return chat_response_to_completion_response(chat_response)

    async def astream_complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        chat_message = self._get_multi_modal_chat_message(
            prompt=prompt,
            role=MessageRole.USER,
            image_documents=image_documents,
        )
        chat_response = await self.astream_chat([chat_message], **kwargs)
        return astream_chat_response_to_completion_response(chat_response)
