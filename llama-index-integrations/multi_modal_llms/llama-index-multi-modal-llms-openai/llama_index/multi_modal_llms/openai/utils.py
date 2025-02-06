import logging
from pathlib import Path
from typing import Optional, Sequence

from llama_index.core.base.llms.types import ImageBlock
from llama_index.core.multi_modal_llms.base import ChatMessage, ImageNode

DEFAULT_OPENAI_API_TYPE = "open_ai"
DEFAULT_OPENAI_API_BASE = "https://api.openai.com/v1"


GPT4V_MODELS = {
    "gpt-4-vision-preview": 128000,
    "gpt-4-turbo-2024-04-09": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4o": 128000,
    "gpt-4o-2024-05-13": 128000,
    "gpt-4o-2024-08-06": 128000,
    "gpt-4o-2024-11-20": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4o-mini-2024-07-18": 128000,
    "o1": 200000,
    "o1-2024-12-17": 200000,
    "o3-mini": 200000,
    "o3-mini-2025-01-31": 200000,
}


MISSING_API_KEY_ERROR_MESSAGE = """No API key found for OpenAI.
Please set either the OPENAI_API_KEY environment variable or \
openai.api_key prior to initialization.
API keys can be found or created at \
https://platform.openai.com/account/api-keys
"""

logger = logging.getLogger(__name__)


def generate_openai_multi_modal_chat_message(
    prompt: str,
    role: str,
    image_documents: Optional[Sequence[ImageNode]] = None,
    image_detail: Optional[str] = "low",
) -> ChatMessage:
    """Create a ChatMessage to be used in a multimodal query."""
    chat_msg = ChatMessage(role=role, content=prompt)
    if image_documents is None:
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
