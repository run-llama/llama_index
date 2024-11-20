"""Unit tests for `llama_index.core.multi_modal_llms.generic_utils`."""

import pytest
import base64
from unittest.mock import mock_open, patch, MagicMock

from llama_index.core.schema import ImageDocument

from llama_index.core.multi_modal_llms.generic_utils import (
    load_image_urls,
    encode_image,
    image_documents_to_base64,
    infer_image_mimetype_from_base64,
    infer_image_mimetype_from_file_path,
    set_base64_and_mimetype_for_image_docs,
)

# Expected values
EXP_IMAGE_URLS = ["http://example.com/image1.jpg"]
EXP_BASE64 = "SGVsbG8gV29ybGQ="  # "Hello World" in base64
EXP_BINARY = b"Hello World"


@pytest.fixture()
def mock_successful_response():
    mock_response = MagicMock()
    mock_response.content = EXP_BINARY
    return mock_response


def test_load_image_urls():
    """Test loading image URLs into ImageDocument objects."""
    result = load_image_urls(EXP_IMAGE_URLS)

    assert len(result) == len(EXP_IMAGE_URLS)
    assert all(isinstance(doc, ImageDocument) for doc in result)
    assert all(doc.image_url == url for doc, url in zip(result, EXP_IMAGE_URLS))


def test_load_image_urls_with_empty_list():
    """Test loading an empty list of URLs."""
    result = load_image_urls([])
    assert result == []


def test_encode_image():
    """Test successful image encoding."""
    with patch("builtins.open", mock_open(read_data=EXP_BINARY)):
        result = encode_image("fake_image.jpg")

    assert result == EXP_BASE64


def test_image_documents_to_base64_multiple_sources():
    """Test converting multiple ImageDocuments with different source types."""
    documents = [
        ImageDocument(image=EXP_BASE64),
        ImageDocument(image_path="test.jpg"),
        ImageDocument(metadata={"file_path": "test.jpg"}),
        ImageDocument(image_url=EXP_IMAGE_URLS[0]),
    ]
    with patch("requests.get") as mock_get:
        mock_get.return_value.content = EXP_BINARY
        with patch("builtins.open", mock_open(read_data=EXP_BINARY)):
            result = image_documents_to_base64(documents)

    assert len(result) == 4
    assert all(encoding == EXP_BASE64 for encoding in result)


def test_image_documents_to_base64_failed_url():
    """Test handling of failed URL requests."""
    document = ImageDocument(image_url=EXP_IMAGE_URLS[0])
    with patch("requests.get"):
        result = image_documents_to_base64([document])

    assert result == []


def test_image_documents_to_base64_empty_sequence():
    """Test handling of empty sequence of documents."""
    result = image_documents_to_base64([])
    assert result == []


def test_image_documents_to_base64_invalid_metadata():
    """Test handling of document with invalid metadata path."""
    document = ImageDocument(metadata={"file_path": ""})
    result = image_documents_to_base64([document])
    assert result == []


def test_complete_workflow():
    """Test the complete workflow from URL to base64 encoding."""
    documents = load_image_urls(EXP_IMAGE_URLS)
    with patch("requests.get") as mock_get:
        mock_get.return_value.content = EXP_BINARY
        result = image_documents_to_base64(documents)

    assert len(result) == len(EXP_IMAGE_URLS)
    assert isinstance(result[0], str)
    assert base64.b64decode(result[0]) == EXP_BINARY


def test_infer_image_mimetype_from_base64():
    """Test inferring image mimetype from base64-encoded data."""
    # Create a minimal valid PNG in base64
    base64_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="

    result = infer_image_mimetype_from_base64(base64_png)
    assert result == "image/png"

    # Valid, meaningless base64
    result = infer_image_mimetype_from_base64(EXP_BASE64)
    assert result is None


def test_infer_image_mimetype_from_file_path():
    """Test inferring image mimetype from file extensions."""
    # JPG/JPEG
    assert infer_image_mimetype_from_file_path("image.jpg") == "image/jpeg"
    assert infer_image_mimetype_from_file_path("image.jpeg") == "image/jpeg"

    # PNG
    assert infer_image_mimetype_from_file_path("image.png") == "image/png"

    # GIF
    assert infer_image_mimetype_from_file_path("image.gif") == "image/gif"

    # WEBP
    assert infer_image_mimetype_from_file_path("image.webp") == "image/webp"

    # Catch-all defaults
    assert infer_image_mimetype_from_file_path("image.asf32") == "image/jpeg"
    assert infer_image_mimetype_from_file_path("") == "image/jpeg"


def test_set_base64_and_mimetype_for_image_docs():
    """Test setting base64 and mimetype fields for ImageDocument objects."""
    image_docs = [
        ImageDocument(image=EXP_BASE64),
        ImageDocument(image_path="test.asdf"),
    ]

    with patch("requests.get") as mock_get:
        mock_get.return_value.content = EXP_BINARY
        with patch("builtins.open", mock_open(read_data=EXP_BINARY)):
            results = set_base64_and_mimetype_for_image_docs(image_docs)

    assert len(results) == 2
    assert results[0].image == EXP_BASE64
    assert results[0].image_mimetype == "image/jpeg"
    assert results[1].image_mimetype == "image/jpeg"
