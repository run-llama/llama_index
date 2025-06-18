import httpx
import pytest
from pathlib import Path

from llama_index.core.schema import ImageNode
from llama_index.core.llms import ImageBlock, TextBlock
from llama_index.multi_modal_llms.gemini.utils import (
    generate_gemini_multi_modal_chat_message,
)

pytest.skip(reason="This package has been deprecated", allow_module_level=True)


@pytest.fixture()
def image_url() -> str:
    return "https://astrabert.github.io/hophop-science/images/whale_doing_science.png"


def test_generate_message_no_image_documents():
    result = generate_gemini_multi_modal_chat_message(
        prompt="Hello", role="user", image_documents=None
    )
    assert result.role == "user"
    assert result.content == "Hello"


def test_generate_message_empty_image_documents():
    result = generate_gemini_multi_modal_chat_message(
        prompt="Hello", role="user", image_documents=[]
    )
    assert result.role == "user"
    assert result.content == "Hello"
    assert result.blocks == [TextBlock(text="Hello")]


def test_generate_message_with_image_documents(tmp_path: Path, image_url: str):
    image_path = tmp_path / "test_image.png"
    image_path.write_bytes(httpx.get(image_url).content)
    image1 = ImageNode(image_path=image_path)
    image2 = ImageNode(image_url=image_url)
    image_documents = [image1, image2]

    result = generate_gemini_multi_modal_chat_message(
        prompt="Hello", role="user", image_documents=image_documents
    )
    assert result.role == "user"
    assert result.content == "Hello"
    assert result.blocks == [
        TextBlock(text="Hello"),
        ImageBlock(path=image_path),
        ImageBlock(url=image_url),
    ]
