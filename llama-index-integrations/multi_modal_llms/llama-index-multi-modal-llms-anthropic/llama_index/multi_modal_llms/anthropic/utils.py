import base64
import logging
from typing import List, Optional, Sequence, Tuple, cast
import filetype

from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from llama_index.core.llms import ImageBlock, ChatMessage, TextBlock
from llama_index.core.schema import ImageDocument, ImageNode
from llama_index.core.base.llms.generic_utils import image_node_to_image_block

DEFAULT_ANTHROPIC_API_TYPE = "anthropic_ai"
DEFAULT_ANTHROPIC_API_BASE = "https://api.anthropic.com"
DEFAULT_ANTHROPIC_API_VERSION = ""


ANTHROPIC_MULTI_MODAL_MODELS = {
    "claude-3-opus-latest": 180000,
    "claude-3-opus-20240229": 180000,
    "claude-3-sonnet-latest": 180000,
    "claude-3-sonnet-20240229": 180000,
    "claude-3-haiku-latest": 180000,
    "claude-3-haiku-20240307": 180000,
    "claude-3-5-sonnet-latest": 180000,
    "claude-3-5-sonnet-20240620": 180000,
    "claude-3-5-sonnet-20241022": 180000,
    "claude-3-5-haiku-20241022": 180000,
}


MISSING_API_KEY_ERROR_MESSAGE = """No API key found for Anthropic.
Please set either the ANTHROPIC_API_KEY environment variable \
API keys can be found or created at \
https://console.anthropic.com/settings/keys
"""

logger = logging.getLogger(__name__)


def infer_image_mimetype_from_base64(base64_string) -> str:
    # Decode the base64 string
    decoded_data = base64.b64decode(base64_string)

    # Use filetype to guess the MIME type
    kind = filetype.guess(decoded_data)

    # Return the MIME type if detected, otherwise return None
    return kind.mime if kind is not None else None


def infer_image_mimetype_from_file_path(image_file_path: str) -> str:
    # Get the file extension
    file_extension = image_file_path.split(".")[-1].lower()

    # Map file extensions to mimetypes
    # Claude 3 support the base64 source type for images, and the image/jpeg, image/png, image/gif, and image/webp media types.
    # https://docs.anthropic.com/claude/reference/messages_post
    if file_extension == "jpg" or file_extension == "jpeg":
        return "image/jpeg"
    elif file_extension == "png":
        return "image/png"
    elif file_extension == "gif":
        return "image/gif"
    elif file_extension == "webp":
        return "image/webp"
    # Add more mappings for other image types if needed

    # If the file extension is not recognized
    return "image/jpeg"


def generate_anthropic_multi_modal_chat_message(
    prompt: str,
    role: str,
    image_documents: Optional[Sequence[ImageDocument]] = None,
) -> List[ChatMessage]:
    # if image_documents is empty, return text only chat message
    if image_documents is None:
        return [ChatMessage.model_validate({"role": role, "content": prompt})]

    # if image_documents is not empty, return text with images chat message
    completion_content = []
    if all(isinstance(doc, ImageNode) for doc in image_documents):
        image_docs: List[ImageBlock] = [
            image_node_to_image_block(doc) for doc in image_documents
        ]
    else:
        image_docs = cast(List[ImageBlock], image_documents)
    blocks = image_docs.extend([TextBlock(text=prompt)])
    return [ChatMessage(role=role, blocks=blocks)]


def resolve_anthropic_credentials(
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
) -> Tuple[Optional[str], str, str]:
    """
    "Resolve Anthropic credentials.

    The order of precedence is:
    1. param
    2. env
    3. anthropic module
    4. default
    """
    # resolve from param or env
    api_key = get_from_param_or_env("api_key", api_key, "ANTHROPIC_API_KEY", "")
    api_base = get_from_param_or_env("api_base", api_base, "ANTHROPIC_API_BASE", "")
    api_version = get_from_param_or_env(
        "api_version", api_version, "ANTHROPIC_API_VERSION", ""
    )

    # resolve from Anthropic module or default
    final_api_key = api_key or ""
    final_api_base = api_base or DEFAULT_ANTHROPIC_API_BASE
    final_api_version = api_version or DEFAULT_ANTHROPIC_API_VERSION

    return final_api_key, str(final_api_base), final_api_version
