import logging
from typing import Any, Dict, Optional, Sequence

from llama_index.legacy.multi_modal_llms.base import ChatMessage
from llama_index.legacy.multi_modal_llms.generic_utils import encode_image
from llama_index.legacy.schema import ImageDocument

DEFAULT_OPENAI_API_TYPE = "open_ai"
DEFAULT_OPENAI_API_BASE = "https://api.openai.com/v1"


GPT4V_MODELS = {
    "gpt-4-vision-preview": 128000,
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
    image_documents: Optional[Sequence[ImageDocument]] = None,
    image_detail: Optional[str] = "low",
) -> ChatMessage:
    # if image_documents is empty, return text only chat message
    if image_documents is None:
        return ChatMessage(role=role, content=prompt)

    # if image_documents is not empty, return text with images chat message
    completion_content = [{"type": "text", "text": prompt}]
    for image_document in image_documents:
        image_content: Dict[str, Any] = {}
        mimetype = image_document.image_mimetype or "image/jpeg"
        if image_document.image and image_document.image != "":
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mimetype};base64,{image_document.image}",
                    "detail": image_detail,
                },
            }
        elif image_document.image_url and image_document.image_url != "":
            image_content = {
                "type": "image_url",
                "image_url": image_document.image_url,
            }
        elif image_document.image_path and image_document.image_path != "":
            base64_image = encode_image(image_document.image_path)
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mimetype};base64,{base64_image}",
                    "detail": image_detail,
                },
            }
        elif (
            "file_path" in image_document.metadata
            and image_document.metadata["file_path"] != ""
        ):
            base64_image = encode_image(image_document.metadata["file_path"])
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": image_detail,
                },
            }

        completion_content.append(image_content)
    return ChatMessage(role=role, content=completion_content)
