import logging
from typing import Any, Dict, List, Sequence

from openai.types.chat import ChatCompletionMessageParam

from llama_index.llms.openai_utils import to_openai_message_dicts
from llama_index.multi_modal_llms.base import ChatMessage
from llama_index.multi_modal_llms.generic_utils import encode_image
from llama_index.schema import ImageDocument

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


def to_openai_multi_modal_payload(
    prompt: str, image_documents: Sequence[ImageDocument], image_detail: str
) -> List[ChatCompletionMessageParam]:
    completion_content = [{"type": "text", "text": prompt}]
    for image_document in image_documents:
        image_content: Dict[str, Any] = {}
        if image_document.image_url and image_document.image_url != "":
            image_content = {
                "type": "image_url",
                "image_url": image_document.image_url,
            }
        elif image_document.image_path and image_document.image_path != "":
            base64_image = encode_image(image_document.image_path)
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
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
    return to_openai_message_dicts(
        [ChatMessage(role="user", content=completion_content)]
    )
