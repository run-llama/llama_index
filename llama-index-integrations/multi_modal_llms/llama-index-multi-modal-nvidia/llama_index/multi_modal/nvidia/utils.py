import base64
import requests
import filetype
from typing import Any, Dict, List, Optional, Sequence
from llama_index.core.schema import ImageDocument

DEFAULT_MODEL = "google/deplot"
BASE_URL = "https://ai.api.nvidia.com/v1/"

KNOWN_URLS = [
    BASE_URL,
    "https://integrate.api.nvidia.com/v1",
]

NVIDIA_MULTI_MODAL_MODELS = {
    "adept/fuyu-8b": {"endpoint": f"{BASE_URL}vlm/adept/fuyu-8b"},
    "google/deplot": {"endpoint": f"{BASE_URL}vlm/google/deplot"},
    "microsoft/kosmos-2": {"endpoint": f"{BASE_URL}vlm/microsoft/kosmos-2"},
    "nvidia/neva-22b": {"endpoint": f"{BASE_URL}vlm/nvidia/neva-22b"},
    "google/paligemma": {"endpoint": f"{BASE_URL}vlm/google/paligemma"},
    "microsoft/phi-3-vision-128k-instruct": {
        "endpoint": f"{BASE_URL}vlm/microsoft/phi-3-vision-128k-instruct"
    },
    "microsoft/phi-3.5-vision-instruct": {
        "endpoint": f"{BASE_URL}microsoft/microsoft/phi-3_5-vision-instruct"
    },
    "nvidia/vila": {"endpoint": f"{BASE_URL}vlm/nvidia/vila"},
}


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
    if file_extension in ["jpg", "jpeg", "png"]:
        return file_extension
    return "png"
    # Add more mappings for other image types if needed

    # If the file extension is not recognized


# Function to encode the image to base64 content
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def create_image_content(image_document) -> Optional[Dict[str, Any]]:
    """
    Create the image content based on the provided image document.
    """
    if image_document.image:
        mimetype = image_document.mimetype if image_document.mimetype else "jpeg"
        return {
            "type": "text",
            "text": f'<img src="data:image/{mimetype};base64,{image_document.image}" />',
        }, ""

    if "asset_id" in image_document.metadata:
        asset_id = image_document.metadata["asset_id"]
        mimetype = image_document.mimetype if image_document.mimetype else "jpeg"
        return {
            "type": "text",
            "text": f'<img src="data:image/{mimetype};asset_id,{asset_id}" />',
        }, asset_id

    if image_document.image_url and image_document.image_url != "":
        mimetype = infer_image_mimetype_from_file_path(image_document.image_url)
        response = requests.get(image_document.image_url)
        try:
            data = base64.b64encode(response.content).decode("utf-8")
            return {
                "type": "text",
                "text": f'<img src="data:image/{mimetype};base64,{data}" />',
            }, ""
        except Exception as e:
            raise "Cannot encode the image url-> {e}"

    return None, None


def generate_nvidia_multi_modal_chat_message(
    prompt: str,
    role: str,
    image_documents: Optional[Sequence[ImageDocument]] = None,
) -> List[Dict[str, Any]]:
    # If image_documents is None, return a text-only chat message
    completion_content = []
    asset_ids = []
    extra_headers = {}

    # Process each image document
    for image_document in image_documents:
        image_content, asset_id = create_image_content(image_document)
        if image_content:
            completion_content.append(image_content)
        if asset_id:
            asset_ids.append(asset_id)

    # Append the text prompt to the completion content
    completion_content.append({"type": "text", "text": prompt})

    if asset_ids:
        extra_headers["NVCF-INPUT-ASSET-REFERENCES"] = ",".join(asset_ids)

    return [{"role": role, "content": completion_content}], extra_headers
