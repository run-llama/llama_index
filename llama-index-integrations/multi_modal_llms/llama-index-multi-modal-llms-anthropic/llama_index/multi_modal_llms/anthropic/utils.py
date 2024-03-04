import logging
from typing import Any, Dict, Optional, Sequence, Tuple, List

from llama_index.core.multi_modal_llms.generic_utils import encode_image
from llama_index.core.schema import ImageDocument
from llama_index.core.base.llms.generic_utils import get_from_param_or_env

DEFAULT_ANTHROPIC_API_TYPE = "anthropic_ai"
DEFAULT_ANTHROPIC_API_BASE = "https://api.anthropic.com"
DEFAULT_ANTHROPIC_API_VERSION = ""


ANTHROPIC_MULTI_MODAL_MODELS = {
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
}


MISSING_API_KEY_ERROR_MESSAGE = """No API key found for Anthropic.
Please set either the ANTHROPIC_API_KEY environment variable \
API keys can be found or created at \
https://console.anthropic.com/settings/keys
"""

logger = logging.getLogger(__name__)


def generate_anthropic_multi_modal_chat_message(
    prompt: str,
    role: str,
    image_documents: Optional[Sequence[ImageDocument]] = None,
) -> List[Dict[str, Any]]:
    # if image_documents is empty, return text only chat message
    if image_documents is None:
        return [{"role": role.value, "content": prompt}]

    # if image_documents is not empty, return text with images chat message
    completion_content = []
    for image_document in image_documents:
        image_content: Dict[str, Any] = {}
        mimetype = image_document.image_mimetype or "image/png"
        if image_document.image_path and image_document.image_path != "":
            base64_image = encode_image(image_document.image_path)
            image_content = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mimetype,
                    "data": base64_image,
                },
            }
        elif (
            "file_path" in image_document.metadata
            and image_document.metadata["file_path"] != ""
        ):
            base64_image = encode_image(image_document.metadata["file_path"])
            image_content = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mimetype,
                    "data": base64_image,
                },
            }

        completion_content.append(image_content)

    completion_content.append({"type": "text", "text": prompt})

    return [{"role": role.value, "content": completion_content}]


def resolve_anthropic_credentials(
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
) -> Tuple[Optional[str], str, str]:
    """ "Resolve OpenAI credentials.

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

    # resolve from openai module or default
    final_api_key = api_key or ""
    final_api_base = api_base or DEFAULT_ANTHROPIC_API_BASE
    final_api_version = api_version or DEFAULT_ANTHROPIC_API_VERSION

    return final_api_key, str(final_api_base), final_api_version
