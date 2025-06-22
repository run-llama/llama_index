import httpx
import pytest

from pathlib import Path
from llama_index.core.schema import ImageDocument


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
