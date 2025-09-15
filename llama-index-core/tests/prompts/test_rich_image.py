import pathlib
from PIL import Image
from llama_index.core.base.llms.types import ImageBlock, ChatMessage
from llama_index.core.prompts import RichPromptTemplate

def _create_temp_jpeg(path: pathlib.Path) -> None:
    img = Image.new("RGB", (8, 8), (255, 0, 0))
    img.save(path, format="JPEG")

def _get_image_block(messages: list[ChatMessage]) -> ImageBlock:
    for b in messages[0].blocks:
        if getattr(b, "block_type", "") == "image":
            return b
    raise AssertionError("No ImageBlock found in messages")

def test_local_path_image_in_bytes(tmp_path: pathlib.Path) -> None:
    img_path = tmp_path / "sample.jpg"
    _create_temp_jpeg(img_path)

    prompt = RichPromptTemplate(
        """
        {% chat role='user' %}
        Describe the following image: {{ image_path | image }}
        {% endchat %}
        """
    )

    messages = prompt.format_messages(image_path=str(img_path))
    block: ImageBlock = _get_image_block(messages)

    assert block.image is not None
    assert block.url is None
    assert block.path is None

def test_http_url() -> None:
    http_url = "https://example.com/img.png"
    prompt = RichPromptTemplate(
        """
        {% chat role='user' %}
        Here is an image: {{ image_path | image }}
        {% endchat %}
        """
    )

    messages = prompt.format_messages(image_path=http_url)
    block = _get_image_block(messages)

    assert str(block.url) == http_url
    assert block.image is None
    assert block.path is None

