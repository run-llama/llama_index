from typing import Any, Optional, Sequence

from llama_index.core.multi_modal_llms.base import ChatMessage
from llama_index.core.schema import ImageDocument


def generate_gemini_multi_modal_chat_message(
    prompt: str,
    role: str,
    image_documents: Optional[Sequence[ImageDocument]] = None,
    **kwargs: Any,
) -> ChatMessage:
    # if image_documents is empty, return text only chat message
    if image_documents is None or len(image_documents) == 0:
        return ChatMessage(role=role, content=prompt)

    additional_kwargs = {
        "images": image_documents,
    }
    return ChatMessage(role=role, content=prompt, additional_kwargs=additional_kwargs)
