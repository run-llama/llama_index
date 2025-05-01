import base64
import os

import filetype
import logging
from typing import List, Optional, Sequence

import requests

from llama_index.core.schema import ImageDocument

logger = logging.getLogger(__name__)


def load_image_urls(image_urls: List[str]) -> List[ImageDocument]:
    """
    Convert a list of image URLs into ImageDocument objects.

    Args:
        image_urls (List[str]): List of strings containing valid image URLs.

    Returns:
        List[ImageDocument]: List of ImageDocument objects.

    """
    return [ImageDocument(image_url=url) for url in image_urls]


def encode_image(image_path: str) -> str:
    """
    Create base64 representation of an image.

    Args:
        image_path (str): Path to the image file

    Returns:
        str: Base64 encoded string of the image

    Raises:
        FileNotFoundError: If the `image_path` doesn't exist.
        IOError: If there's an error reading the file.

    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_documents_to_base64(
    image_documents: Sequence[ImageDocument],
) -> List[str]:
    """
    Convert ImageDocument objects to base64-encoded strings.

    Args:
        image_documents (Sequence[ImageDocument]: Sequence of
            ImageDocument objects

    Returns:
        List[str]: List of base64-encoded image strings

    """
    image_encodings = []

    # Encode image documents to base64
    for image_document in image_documents:
        if image_document.image:  # This field is already base64-encoded
            image_encodings.append(image_document.image)
        elif image_document.image_path and os.path.isfile(
            image_document.image_path
        ):  # This field is a path to the image, which is then encoded.
            image_encodings.append(encode_image(image_document.image_path))
        elif (
            "file_path" in image_document.metadata
            and image_document.metadata["file_path"] != ""
            and os.path.isfile(image_document.metadata["file_path"])
        ):  # Alternative path to the image, which is then encoded.
            image_encodings.append(encode_image(image_document.metadata["file_path"]))
        elif image_document.image_url:  # Image can also be pulled from the URL.
            response = requests.get(image_document.image_url)
            try:
                image_encodings.append(
                    base64.b64encode(response.content).decode("utf-8")
                )
            except Exception as e:
                logger.warning(f"Cannot encode the image pulled from URL -> {e}")
    return image_encodings


def infer_image_mimetype_from_file_path(image_file_path: str) -> str:
    """
    Infer the MIME of an image file based on its file extension.

    Currently only supports the following types of images:
        * image/jpeg
        * image/png
        * image/gif
        * image/webp

    Args:
        image_file_path (str): Path to the image file.

    Returns:
        str: MIME type of the image: image/jpeg, image/png, image/gif, or image/webp.
            Defaults to `image/jpeg`.

    """
    # Get the file extension
    file_extension = image_file_path.split(".")[-1].lower()

    # Map file extensions to mimetypes
    if file_extension == "jpg" or file_extension == "jpeg":
        return "image/jpeg"
    elif file_extension == "png":
        return "image/png"
    elif file_extension == "gif":
        return "image/gif"
    elif file_extension == "webp":
        return "image/webp"

    # If the file extension is not recognized
    return "image/jpeg"


def infer_image_mimetype_from_base64(base64_string: str) -> Optional[str]:
    """
    Infer the MIME of an image from the base64 encoding.

    Args:
        base64_string (str): Base64-encoded string of the image.

    Returns:
        Optional[str]: MIME type of the image: image/jpeg, image/png, image/gif, or image/webp.
          `None` if the MIME type cannot be inferred.

    """
    # Decode the base64 string
    decoded_data = base64.b64decode(base64_string)

    # Use filetype to guess the MIME type
    kind = filetype.guess(decoded_data)

    # Return the MIME type if detected, otherwise return None
    return kind.mime if kind is not None else None


def set_base64_and_mimetype_for_image_docs(
    image_documents: Sequence[ImageDocument],
) -> Sequence[ImageDocument]:
    """
    Set the base64 and mimetype fields for the image documents.

    Args:
        image_documents (Sequence[ImageDocument]): Sequence of ImageDocument objects.

    Returns:
        Sequence[ImageDocument]: ImageDocuments with base64 and detected mimetypes set.

    """
    base64_strings = image_documents_to_base64(image_documents)
    for image_doc, base64_str in zip(image_documents, base64_strings):
        image_doc.image = base64_str
        image_doc.image_mimetype = infer_image_mimetype_from_base64(image_doc.image)
        if not image_doc.image_mimetype and image_doc.image_path:
            image_doc.image_mimetype = infer_image_mimetype_from_file_path(
                image_doc.image_path
            )
        else:
            # Defaults to `image/jpeg` if the mimetype cannot be inferred
            image_doc.image_mimetype = "image/jpeg"
    return image_documents
