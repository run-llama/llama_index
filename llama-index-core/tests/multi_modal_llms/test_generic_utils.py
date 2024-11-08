"""Unit tests for `llama_index.core.multi_modal_llms.generic_utils`."""

import pytest
import base64
from unittest.mock import mock_open, patch, MagicMock
from typing import List

from llama_index.core.schema import ImageDocument

from llama_index.core.multi_modal_llms.generic_utils import (
    load_image_urls,
    encode_image,
    image_documents_to_base64,
)

# Expected values
EXP_IMAGE_URLS = ["http://example.com/image1.jpg"]

EXP_BASE64 = "SGVsbG8gV29ybGQ="  # "Hello World" in base64
EXP_BINARY = b"Hello World"


# Fixtures
@pytest.fixture()
def sample_image_documents() -> List[ImageDocument]:
    return [ImageDocument(image_url=url) for url in EXP_IMAGE_URLS]


@pytest.fixture()
def mock_successful_response():
    mock_response = MagicMock()
    mock_response.content = EXP_BINARY
    return mock_response


def test_load_image_urls_with_valid_urls():
    """Test loading valid image URLs into ImageDocument objects."""
    result = load_image_urls(EXP_IMAGE_URLS)

    assert len(result) == len(EXP_IMAGE_URLS)
    assert all(isinstance(doc, ImageDocument) for doc in result)
    assert all(doc.image_url == url for doc, url in zip(result, EXP_IMAGE_URLS))


def test_load_image_urls_with_empty_list():
    """Test loading an empty list of URLs."""
    result = load_image_urls([])
    assert result == []


# Tests for encode_image
def test_encode_image_successful():
    """Test successful image encoding."""
    mock_file_content = b"image content"
    expected_b64 = base64.b64encode(mock_file_content).decode("utf-8")

    with patch("builtins.open", mock_open(read_data=mock_file_content)):
        result = encode_image("fake_image.jpg")

    assert result == expected_b64


def test_encode_image_file_not_found():
    """Test handling of non-existent image file."""
    with patch("builtins.open", mock_open()) as mock_file:
        mock_file.side_effect = FileNotFoundError()
        with pytest.raises(FileNotFoundError):
            encode_image("nonexistent.jpg")


def test_encode_image_io_error():
    """Test handling of IO error during image reading."""
    with patch("builtins.open", mock_open()) as mock_file:
        mock_file.side_effect = OSError()
        with pytest.raises(IOError):
            encode_image("corrupted.jpg")


@pytest.mark.parametrize(
    ("image_document", "expected_result"),
    [
        (ImageDocument(image=EXP_BASE64), [EXP_BASE64]),
        (ImageDocument(image_path="test.jpg"), [EXP_BASE64]),
        (ImageDocument(metadata={"file_path": "test.jpg"}), [EXP_BASE64]),
        (ImageDocument(image_url="http://example.com/image.jpg"), [EXP_BASE64]),
    ],
)
def test_image_documents_to_base64_single_document(
    image_document, expected_result, mock_successful_response
):
    """Test converting single ImageDocument with different configurations."""
    with patch("requests.get", return_value=mock_successful_response):
        with patch(
            "llama_index.core.multi_modal_llms.generic_utils.encode_image",
            return_value=EXP_BASE64,
        ):
            result = image_documents_to_base64([image_document])
            assert result == expected_result


def test_image_documents_to_base64_multiple_sources():
    """Test converting multiple ImageDocuments with different source types."""
    documents = [
        ImageDocument(image=EXP_BASE64),
        ImageDocument(image_path="test.jpg"),
        ImageDocument(metadata={"file_path": "test.jpg"}),
        ImageDocument(image_url="http://example.com/image.jpg"),
    ]

    with patch("requests.get") as mock_get:
        mock_get.return_value.content = EXP_BINARY
        with patch(
            "llama_index.core.multi_modal_llms.generic_utils.encode_image",
            return_value=EXP_BASE64,
        ):
            result = image_documents_to_base64(documents)

            assert len(result) == 4
            assert all(encoding == EXP_BASE64 for encoding in result)


def test_image_documents_to_base64_failed_url():
    """Test handling of failed URL requests."""
    document = ImageDocument(image_url="http://example.com/bad_image.jpg")

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


# Integration-style tests
def test_complete_workflow():
    """Test the complete workflow from URL to base64 encoding."""
    urls = ["http://example.com/test.jpg"]
    documents = load_image_urls(urls)

    with patch("requests.get") as mock_get:
        mock_get.return_value.content = EXP_BINARY
        result = image_documents_to_base64(documents)

        assert len(result) == 1
        assert isinstance(result[0], str)
        assert base64.b64decode(result[0].encode("utf-8"))
