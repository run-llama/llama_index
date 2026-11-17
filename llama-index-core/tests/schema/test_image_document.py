import httpx
import pytest

from pathlib import Path, PureWindowsPath
from llama_index.core.schema import ImageDocument, MediaResource


@pytest.fixture()
def image_url() -> str:
    return "https://astrabert.github.io/hophop-science/images/whale_doing_science.png"


def test_real_image_path(tmp_path: Path, image_url: str) -> None:
    content = httpx.get(image_url).content
    fl_path = tmp_path / "test_image.png"
    fl_path.write_bytes(content)
    doc = ImageDocument(image_path=fl_path.__str__())
    assert isinstance(doc, ImageDocument)


def test_real_image_url(image_url: str) -> None:
    doc = ImageDocument(image_url=image_url)
    assert isinstance(doc, ImageDocument)


def test_non_image_path(tmp_path: Path) -> None:
    fl_path = tmp_path / "test_file.txt"
    fl_path.write_text("Hello world!")
    with pytest.raises(expected_exception=ValueError):
        doc = ImageDocument(image_path=fl_path.__str__())


def test_non_image_url(image_url: str) -> None:
    image_url = image_url.replace("png", "txt")
    with pytest.raises(expected_exception=ValueError):
        doc = ImageDocument(image_url=image_url)


def test_image_path_is_posix_on_every_platform() -> None:
    """
    The image_path accessor must not embed the platform path separator.

    Mirrors test_serialize_path_is_posix_on_every_platform in
    test_media_resource.py: PureWindowsPath stands in for a WindowsPath, which
    cannot be instantiated on POSIX.
    """
    doc = ImageDocument.model_construct(
        image_resource=MediaResource.model_construct(path=PureWindowsPath("a/b/c.txt"))
    )
    assert doc.image_path == "a/b/c.txt"
