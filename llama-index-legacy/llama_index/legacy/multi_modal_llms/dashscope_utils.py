"""DashScope api utils."""

from http import HTTPStatus
from typing import Any, Dict, List, Sequence

from llama_index.legacy.core.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
)
from llama_index.legacy.schema import ImageDocument


def dashscope_response_to_completion_response(response: Any) -> CompletionResponse:
    if response["status_code"] == HTTPStatus.OK:
        content = response["output"]["choices"][0]["message"]["content"]
        if content:
            content = content[0]["text"]
        else:
            content = ""
        return CompletionResponse(text=content, raw=response)
    else:
        return CompletionResponse(text="", raw=response)


def dashscope_response_to_chat_response(
    response: Any,
) -> ChatResponse:
    if response["status_code"] == HTTPStatus.OK:
        content = response["output"]["choices"][0]["message"]["content"]
        role = response["output"]["choices"][0]["message"]["role"]
        return ChatResponse(
            message=ChatMessage(role=role, content=content), raw=response
        )
    else:
        return ChatResponse(message=ChatMessage(), raw=response)


def chat_message_to_dashscope_multi_modal_messages(
    chat_messages: Sequence[ChatMessage],
) -> List[Dict]:
    messages = []
    for msg in chat_messages:
        messages.append({"role": msg.role.value, "content": msg.content})
    return messages


def create_dashscope_multi_modal_chat_message(
    prompt: str, role: str, image_documents: Sequence[ImageDocument]
) -> ChatMessage:
    if image_documents is None:
        message = ChatMessage(role=role, content=[{"text": prompt}])
    else:
        content = []
        for image_document in image_documents:
            content.append(
                {
                    "image": (
                        image_document.image_url
                        if image_document.image_url is not None
                        else image_document.image_path
                    )
                }
            )
        content.append({"text": prompt})
        message = ChatMessage(role=role, content=content)

    return message


def load_local_images(local_images: List[str]) -> List[ImageDocument]:
    # load images into image documents
    image_documents = []
    for _, img in enumerate(local_images):
        new_image_document = ImageDocument(image_path=img)
        image_documents.append(new_image_document)
    return image_documents
