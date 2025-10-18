from typing import Any, Sequence, Union

from llama_index.core.base.llms.generic_utils import (
    chat_response_to_completion_response,
    stream_chat_response_to_completion_response,
    astream_chat_response_to_completion_response,
    image_node_to_image_block,
)
from llama_index.core.base.llms.types import (
    ChatMessage,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    MessageRole,
    ImageBlock,
    TextBlock,
)
from llama_index.core.schema import ImageNode
from llama_index.llms.fireworks import Fireworks


class FireworksMultiModal(Fireworks):
    def __init__(
        self,
        model: str = "accounts/fireworks/models/phi-3-vision-128k-instruct",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "fireworks_multi_modal_llms"

    def _get_multi_modal_chat_message(
        self,
        prompt: str,
        role: str,
        image_documents: Sequence[Union[ImageBlock, ImageNode]],
    ) -> ChatMessage:
        chat_msg = ChatMessage(role=role, content=prompt)
        if not image_documents:
            # if image_documents is empty, return text only chat message
            return chat_msg

        chat_msg.blocks.append(TextBlock(text=prompt))

        if all(isinstance(doc, ImageNode) for doc in image_documents):
            chat_msg.blocks.extend(
                [image_node_to_image_block(doc) for doc in image_documents]
            )
        else:
            chat_msg.blocks.extend(image_documents)

        return chat_msg

    def complete(
        self,
        prompt: str,
        image_documents: Sequence[Union[ImageNode, ImageBlock]],
        **kwargs: Any,
    ) -> CompletionResponse:
        chat_message = self._get_multi_modal_chat_message(
            prompt=prompt,
            role=MessageRole.USER.value,
            image_documents=image_documents,
        )
        chat_response = self.chat([chat_message], **kwargs)
        return chat_response_to_completion_response(chat_response)

    def stream_complete(
        self,
        prompt: str,
        image_documents: Sequence[Union[ImageNode, ImageBlock]],
        **kwargs: Any,
    ) -> CompletionResponseGen:
        chat_message = self._get_multi_modal_chat_message(
            prompt=prompt,
            role=MessageRole.USER.value,
            image_documents=image_documents,
        )
        chat_response = self.stream_chat([chat_message], **kwargs)
        return stream_chat_response_to_completion_response(chat_response)

    async def acomplete(
        self,
        prompt: str,
        image_documents: Sequence[Union[ImageNode, ImageBlock]],
        **kwargs: Any,
    ) -> CompletionResponse:
        chat_message = self._get_multi_modal_chat_message(
            prompt=prompt,
            role=MessageRole.USER.value,
            image_documents=image_documents,
        )
        chat_response = await self.achat([chat_message], **kwargs)
        return chat_response_to_completion_response(chat_response)

    async def astream_complete(
        self,
        prompt: str,
        image_documents: Sequence[Union[ImageNode, ImageBlock]],
        **kwargs: Any,
    ) -> CompletionResponseAsyncGen:
        chat_message = self._get_multi_modal_chat_message(
            prompt=prompt,
            role=MessageRole.USER.value,
            image_documents=image_documents,
        )
        chat_response = await self.astream_chat([chat_message], **kwargs)
        return astream_chat_response_to_completion_response(chat_response)
