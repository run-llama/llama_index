import base64
import logging
from typing import List, Sequence

import requests

from llama_index.core.schema import ImageDocument

logger = logging.getLogger(__name__)


def load_image_urls(image_urls: List[str]) -> List[ImageDocument]:
    # load remote image urls into image documents
    image_documents = []
    for i in range(len(image_urls)):
        new_image_document = ImageDocument(image_url=image_urls[i])
        image_documents.append(new_image_document)
    return image_documents


# Function to encode the image to base64 content
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Supporting Ollama like Multi-Modal images base64 encoding
def image_documents_to_base64(
    image_documents: Sequence[ImageDocument],
) -> List[str]:
    image_encodings = []
    # encode image documents to base64
    for image_document in image_documents:
        if image_document.image:
            image_encodings.append(image_document.image)
        elif image_document.image_path:
            image_encodings.append(encode_image(image_document.image_path))
        elif (
            "file_path" in image_document.metadata
            and image_document.metadata["file_path"] != ""
        ):
            image_encodings.append(encode_image(image_document.metadata["file_path"]))
        elif image_document.image_url:
            response = requests.get(image_document.image_url)
            try:
                image_encodings.append(
                    base64.b64encode(response.content).decode("utf-8")
                )
            except Exception as e:
                logger.warning(f"Cannot encode the image url-> {e}")
    return image_encodings
