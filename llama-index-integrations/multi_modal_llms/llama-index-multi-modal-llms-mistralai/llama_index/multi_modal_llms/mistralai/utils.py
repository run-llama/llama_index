import base64
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple
import filetype

from llama_index.core.base.llms.generic_utils import get_from_param_or_env
from llama_index.core.multi_modal_llms.generic_utils import encode_image
from llama_index.core.schema import ImageDocument

DEFAULT_MISTRALAI_API_TYPE = "mistral_ai"
DEFAULT_MISTRALAI_API_BASE = "https://api.mistral.ai/"
DEFAULT_MISTRALAI_API_VERSION = ""


MISTRALAI_MULTI_MODAL_MODELS = {
    "pixtral-12b-2409": 128000,
    "pixtral-large-latest": 128000,
}


MISSING_API_KEY_ERROR_MESSAGE = """No API key found for Mistral.
Please set either the MISTRAL_API_KEY environment variable \
API keys can be found or created at \
https://console.mistral.ai/api-keys/
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
    # Pixtral support the base64 source type for images, and the image/jpeg, image/png, image/gif, and image/webp media types.
    # https://docs.mistral.ai/capabilities/vision/
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


def generate_mistral_multi_modal_chat_message(
    prompt: str,
    role: str,
    image_documents: Optional[Sequence[ImageDocument]] = None,
) -> List[Dict[str, Any]]:
    # if image_documents is empty, return text only chat message
    if image_documents is None:
        return [{"role": role, "content": prompt}]

    # if image_documents is not empty, return text with images chat message
    completion_content = []
    for image_document in image_documents:
        image_content: Dict[str, Any] = {}
        if image_document.image_path and image_document.image_path != "":
            mimetype = infer_image_mimetype_from_file_path(image_document.image_path)
            base64_image = encode_image(image_document.image_path)
            image_content = {
                "type": "image_url",
                "image_url": f"data:{mimetype};base64,{base64_image}",
            }
        elif (
            "file_path" in image_document.metadata
            and image_document.metadata["file_path"] != ""
        ):
            mimetype = infer_image_mimetype_from_file_path(
                image_document.metadata["file_path"]
            )
            base64_image = encode_image(image_document.metadata["file_path"])
            image_content = {
                "type": "image_url",
                "image_url": f"data:{mimetype};base64,{base64_image}",
            }
        elif image_document.image_url and image_document.image_url != "":
            mimetype = infer_image_mimetype_from_file_path(image_document.image_url)
            image_content = {
                "type": "image_url",
                "image_url": f"{image_document.image_url}",
            }
        elif image_document.image != "":
            base64_image = image_document.image
            mimetype = infer_image_mimetype_from_base64(base64_image)
            image_content = {
                "type": "image_url",
                "image_url": f"data:{mimetype};base64,{base64_image}",
            }
        completion_content.append(image_content)

    completion_content.append({"type": "text", "text": prompt})

    return [{"role": role, "content": completion_content}]


def resolve_mistral_credentials(
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
) -> Tuple[Optional[str], str, str]:
    """
    "Resolve Mistral credentials.

    The order of precedence is:
    1. param
    2. env
    3. mistral module
    4. default
    """
    # resolve from param or env
    api_key = get_from_param_or_env("api_key", api_key, "MISTRAL_API_KEY", "")
    api_base = get_from_param_or_env("api_base", api_base, "MISTRAL_API_BASE", "")
    api_version = get_from_param_or_env(
        "api_version", api_version, "MISTRAL_API_VERSION", ""
    )

    # resolve from Mistral module or default
    final_api_key = api_key or ""
    final_api_base = api_base or DEFAULT_MISTRALAI_API_BASE
    final_api_version = api_version or DEFAULT_MISTRALAI_API_VERSION

    return final_api_key, str(final_api_base), final_api_version
