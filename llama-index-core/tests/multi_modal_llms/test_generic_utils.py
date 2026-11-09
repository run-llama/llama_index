"""Unit tests for `llama_index.core.multi_modal_llms.generic_utils`."""

import pytest
import base64
from pathlib import Path
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
EXP_IMAGE_URLS = [
    "https://astrabert.github.io/hophop-science/images/whale_doing_science.png"
]
EXP_BASE64 = "SGVsbG8gV29ybGQ="  # "Hello World" in base64
EXP_BINARY = b"Hello World"


@pytest.fixture(autouse=True)
def validate_image_urls_locally():
    """Keep ImageDocument construction from validating test URLs over the network."""
    with patch("llama_index.core.schema.is_image_url_pil", return_value=True):
        yield


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


def test_image_documents_to_base64_multiple_sources(tmp_path: Path):
    """Test converting multiple ImageDocuments with different source types."""
    from PIL import Image

    fl_path = tmp_path / "test_image.png"
    Image.new("RGB", (1, 1)).save(fl_path)
    content = fl_path.read_bytes()
    expected_b64 = base64.b64encode(content).decode("utf-8")
    documents = [
        ImageDocument(image=expected_b64),
        ImageDocument(image_path=fl_path),
        ImageDocument(metadata={"file_path": "test.jpg"}),
        ImageDocument(image_url=EXP_IMAGE_URLS[0]),
    ]
    with patch("requests.get") as mock_get:
        mock_get.return_value.content = content
        with patch("os.path.isfile", return_value=True):
            with patch("builtins.open", mock_open(read_data=content)):
                result = image_documents_to_base64(documents)

    assert len(result) == 4
    assert all(encoding == expected_b64 for encoding in result)


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


@pytest.mark.parametrize("image_format", ["PNG", "GIF", "WEBP", "JPEG"])
@pytest.mark.parametrize("source", ["inline", "path", "metadata"])
def test_set_base64_and_mimetype_for_image_docs(
    tmp_path: Path, image_format: str, source: str
) -> None:
    """Preserve the format detected from real image content."""
    from PIL import Image

    path = tmp_path / "image.bin"
    Image.new("RGB", (1, 1)).save(path, format=image_format)
    expected_b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    if source == "inline":
        document = ImageDocument(image=expected_b64)
    elif source == "path":
        document = ImageDocument(image_path=str(path))
    else:
        document = ImageDocument(metadata={"file_path": str(path)})
    documents = [document]

    results = set_base64_and_mimetype_for_image_docs(documents)

    assert results is documents
    assert results[0] is document
    assert document.image == expected_b64
    assert document.image_mimetype == f"image/{image_format.lower()}"


def test_set_base64_and_mimetype_uses_default() -> None:
    """Use the default when content detection fails and no path is available."""
    document = ImageDocument(image=EXP_BASE64)

    set_base64_and_mimetype_for_image_docs([document])

    assert document.image == EXP_BASE64
    assert document.image_mimetype == "image/jpeg"
