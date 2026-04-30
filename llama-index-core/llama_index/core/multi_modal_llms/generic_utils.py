import base64
import os
import warnings
from pathlib import Path
from typing import List, Optional, Sequence

import filetype
import logging

import requests

from llama_index.core.schema import ImageDocument

logger = logging.getLogger(__name__)

# Security: Maximum allowed file size (50 MB)
_MAX_FILE_SIZE = 50 * 1024 * 1024

# Security: Allowed image MIME types
_ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/bmp",
    "image/tiff",
}


def _validate_image_path(file_path: str) -> Path:
    """Validate that a file path is safe to read as an image.

    Security checks:
    1. Reject symlinks (prevents symlink-based path traversal)
    2. Validate file is a regular file (not a device, socket, etc.)
    3. Validate file size is within limits (prevents DoS)
    4. Validate file content is actually an image (MIME check)
    5. Reject paths containing null bytes (prevents null-byte injection)

    Args:
        file_path: The file path to validate.

    Returns:
        Resolved Path object if validation passes.

    Raises:
        ValueError: If the path fails any security check.
        FileNotFoundError: If the file does not exist.
    """
    # Reject null bytes in path
    if "\x00" in file_path:
        raise ValueError("File path contains null bytes")

    path = Path(file_path)

    # Check for symlinks before resolving
    if path.is_symlink():
        raise ValueError(
            f"Symlinks are not allowed for image paths: {file_path}. "
            "Use a direct path to the image file instead."
        )

    # Resolve the path (resolves any .. or . components)
    resolved = path.resolve()

    # Check that the resolved path exists and is a file
    if not resolved.is_file():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    # Check file size
    file_size = resolved.stat().st_size
    if file_size > _MAX_FILE_SIZE:
        raise ValueError(
            f"Image file too large: {file_size} bytes (max: {_MAX_FILE_SIZE}). "
            "Files larger than 50 MB are not allowed."
        )

    if file_size == 0:
        raise ValueError(f"Image file is empty: {file_path}")

    # Validate MIME type by reading file header
    try:
        with open(resolved, "rb") as f:
            header = f.read(min(512, file_size))
        kind = filetype.guess(header)
        if kind is None or kind.mime not in _ALLOWED_MIME_TYPES:
            detected = kind.mime if kind is not None else "unknown"
            raise ValueError(
                f"File does not appear to be a valid image: {file_path}. "
                f"Detected type: {detected}. "
                f"Supported formats: {', '.join(sorted(_ALLOWED_MIME_TYPES))}"
            )
    except (OSError, IOError) as e:
        raise ValueError(f"Cannot read image file {file_path}: {e}")

    return resolved


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
        ValueError: If the file is not a valid image or fails security checks.
        IOError: If there is an error reading the file.
    """
    # Security: Validate the path before reading
    resolved_path = _validate_image_path(image_path)
    with open(resolved_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_documents_to_base64(
    image_documents: Sequence[ImageDocument],
) -> List[str]:
    """
    Convert ImageDocument objects to base64-encoded strings.

    Security: All file paths are validated to prevent arbitrary file read
    vulnerabilities. See _validate_image_path() for details.

    Args:
        image_documents (Sequence[ImageDocument]: Sequence of
            ImageDocument objects

    Returns:
        List[str]: List of base64-encoded image strings

    Raises:
        ValueError: If any image path fails security validation.
    """
    image_encodings = []

    # Encode image documents to base64
    for image_document in image_documents:
        if image_document.image:  # This field is already base64-encoded
            image_encodings.append(image_document.image)
        elif image_document.image_path:
            # Security: Validate path before encoding
            try:
                _validate_image_path(image_document.image_path)
                image_encodings.append(encode_image(image_document.image_path))
            except (ValueError, FileNotFoundError) as e:
                logger.warning(
                    f"Skipping image_path due to validation failure: {e}"
                )
        elif (
            "file_path" in image_document.metadata
            and image_document.metadata["file_path"] != ""
        ):
            # DEPRECATION WARNING: metadata["file_path"] is deprecated
            # Use image_path parameter instead, which goes through validation
            warnings.warn(
                "Using metadata['file_path'] for image paths is deprecated. "
                "Please use ImageDocument(image_path=...) instead, which includes "
                "security validation. Support for metadata['file_path'] will be "
                "removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Security: Validate the metadata path too
            try:
                _validate_image_path(image_document.metadata["file_path"])
                image_encodings.append(
                    encode_image(image_document.metadata["file_path"])
                )
            except (ValueError, FileNotFoundError) as e:
                logger.warning(
                    f"Skipping metadata file_path due to validation failure: {e}"
                )
        elif image_document.image_url:  # Image can also be pulled from the URL.
            try:
                response = requests.get(
                    image_document.image_url, timeout=(60, 60)
                )
                response.raise_for_status()
                # Validate that the response is actually an image
                kind = filetype.guess(response.content)
                if kind is None or kind.mime not in _ALLOWED_MIME_TYPES:
                    logger.warning(
                        f"URL does not point to a valid image: "
                        f"{image_document.image_url} "
                        f"(detected: {kind.mime if kind else 'unknown'})"
                    )
                    continue
                image_encodings.append(
                    base64.b64encode(response.content).decode("utf-8")
                )
            except requests.RequestException as e:
                logger.warning(
                    f"Cannot fetch image from URL {image_document.image_url}: {e}"
                )
            except Exception as e:
                logger.warning(
                    f"Cannot encode the image pulled from URL -> {e}"
                )
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
