"""Google's Gemini multi-modal models."""

from typing import Any, Sequence

from llama_index.core.base.llms.types import (
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    MessageRole,
)
from llama_index.core.schema import ImageDocument, ImageNode
from llama_index.llms.gemini import Gemini

from .utils import generate_gemini_multi_modal_chat_message


class GeminiMultiModal(Gemini):
    """
    Gemini multimodal.

    This class is a thin wrapper around Gemini to support legacy multimodal completion methods.
    """

    @classmethod
    def class_name(cls) -> str:
        return "Gemini_MultiModal_LLM"

    def complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponse:
        documents = []
        for node in image_documents:
            data = node.resolve_image().read()
            doc = ImageDocument(image=data)
            documents.append(doc)
        msg = generate_gemini_multi_modal_chat_message(
            prompt=prompt, role=MessageRole.USER, image_documents=documents
        )
        response = self.chat(messages=[msg], **kwargs)
        return CompletionResponse(text=response.message.content or "")

    async def acomplete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponse:
        documents = []
        for node in image_documents:
            data = node.resolve_image().read()
            doc = ImageDocument(image=data)
            documents.append(doc)
        msg = generate_gemini_multi_modal_chat_message(
            prompt=prompt, role=MessageRole.USER, image_documents=documents
        )
        response = await self.achat(messages=[msg], **kwargs)
        return CompletionResponse(text=response.message.content or "")

    def stream_complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponseGen:
        documents = []
        for node in image_documents:
            data = node.resolve_image().read()
            doc = ImageDocument(image=data)
            documents.append(doc)

        msg = generate_gemini_multi_modal_chat_message(
            prompt=prompt, role=MessageRole.USER, image_documents=documents
        )

        def gen() -> CompletionResponseGen:
            for s in self.stream_chat(messages=[msg], **kwargs):
                yield CompletionResponse(
                    text=s.message.content or "", delta=s.delta or ""
                )

        return gen()

    async def astream_complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        documents = []
        for node in image_documents:
            data = node.resolve_image().read()
            doc = ImageDocument(image=data)
            documents.append(doc)

        msg = generate_gemini_multi_modal_chat_message(
            prompt=prompt, role=MessageRole.USER, image_documents=documents
        )

        async def gen() -> CompletionResponseAsyncGen:
            streaming_handler = await self.astream_chat(messages=[msg], **kwargs)
            async for chunk in streaming_handler:
                yield CompletionResponse(
                    text=chunk.message.content or "", delta=chunk.delta or ""
                )

        return gen()
