import base64
import httpx
import platformdirs  # noqa: F401
from datetime import datetime

import pytest
from llama_index.core.prompts.rich import RichPromptTemplate
from llama_index.core.schema import TextNode

from llama_index.core.base.llms.types import (
    TextBlock,
    ImageBlock,
    AudioBlock,
    VideoBlock,
    DocumentBlock,
    ChatMessage,
)


@pytest.fixture()
def png_1px_b64() -> bytes:
    return b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


@pytest.fixture()
def png_1px(png_1px_b64) -> bytes:
    return base64.b64decode(png_1px_b64)


@pytest.fixture()
def pdf_url() -> str:
    return "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"


@pytest.fixture()
def mock_pdf_bytes(pdf_url) -> bytes:
    """
    Returns a byte string representing a very simple, minimal PDF file.
    """
    return httpx.get(pdf_url).content


@pytest.fixture()
def mp3_bytes() -> bytes:
    """
    Small mp3 file bytes (0.2 seconds of audio).
    """
    return b"ID3\x04\x00\x00\x00\x00\x01\tTXXX\x00\x00\x00\x12\x00\x00\x03major_brand\x00isom\x00TXXX\x00\x00\x00\x13\x00\x00\x03minor_version\x00512\x00TXXX\x00\x00\x00$\x00\x00\x03compatible_brands\x00isomiso2avc1mp41\x00TSSE\x00\x00\x00\x0e\x00\x00\x03Lavf62.3.100\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xf3X\xc0\x00\x00\x00\x00\x00\x00\x00\x00\x00Info\x00\x00\x00\x0f\x00\x00\x00\x06\x00\x00\x03<\x00YYYYYYYYYYYYYYYYzzzzzzzzzzzzzzzzz\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\x00\x00\x00\x00Lavf\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00$\x00\x00\x00\x00\x00\x00\x00\x00\x03<\xa6\xbc`\x8e\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xf38\xc4\x00\x00\x00\x03H\x00\x00\x00\x00LAME3.100UUUUUUUUUUUUUUUUUUUUULAME3.100UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\xff\xf38\xc4_\x00\x00\x03H\x00\x00\x00\x00UUUUUUUUUUUUUUUUUUUUUUUUUUUUUULAME3.100UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\xff\xf38\xc4\xa0\x00\x00\x03H\x00\x00\x00\x00UUUUUUUUUUUUUUUUUUUUUUUUUUUUUULAME3.100UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\xff\xf38\xc4\xa0\x00\x00\x03H\x00\x00\x00\x00UUUUUUUUUUUUUUUUUUUUUUUUUUUUUULAME3.100UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\xff\xf38\xc4\xa0\x00\x00\x03H\x00\x00\x00\x00UUUUUUUUUUUUUUUUUUUUUUUUUUUUUULAME3.100UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\xff\xf38\xc4\xa0\x00\x00\x03H\x00\x00\x00\x00UUUUUUUUUUUUUUUUUUUUUUUUUUUUUULAME3.100UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU"


@pytest.fixture()
def mp4_bytes() -> bytes:
    # Minimal fake MP4 header bytes (ftyp box)
    return b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"


def test_basic_rich_prompt():
    prompt = RichPromptTemplate("Hello, {{name}}!")

    assert not prompt.is_chat_template
    assert prompt.template_vars == ["name"]

    formatted_prompt = prompt.format(name="John")
    assert formatted_prompt == "Hello, John!"

    formatted_prompt = prompt.format(name="Jane")
    assert formatted_prompt == "Hello, Jane!"


def test_basic_rich_chat_prompt():
    prompt = RichPromptTemplate("{% chat role='user' %}Hello, {{name}}!{% endchat %}")

    assert prompt.is_chat_template

    messages = prompt.format_messages(name="John")
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == "Hello, John!"


def test_function_mapping():
    def today(**prompt_args):
        return datetime.now().strftime("%Y-%m-%d")

    prompt = RichPromptTemplate(
        "Hello, {{name}}, today is {{today}}", function_mappings={"today": today}
    )

    messages = prompt.format_messages(name="John")
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == "Hello, John, today is " + datetime.now().strftime(
        "%Y-%m-%d"
    )


def test_object_mapping():
    nodes = [
        TextNode(text="You are a helpful assistant."),
        TextNode(text="You are new to the company."),
        TextNode(text="You are a great assistant."),
    ]
    prompt_str = """
Hello, {{name}}. Here is some information about you:

{% for node in nodes %}
- {{node.text}}
{% endfor %}
"""

    prompt = RichPromptTemplate(prompt_str)

    messages = prompt.format_messages(name="John", nodes=nodes)
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert nodes[0].text in messages[0].content
    assert nodes[1].text in messages[0].content
    assert nodes[2].text in messages[0].content


def test_prompt_with_images():
    image_url = "https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png"

    prompt_str = """
{% chat role='user' %}
Hello, {{name}}. Here is an image of you:

{{ your_image | image }}

{% endchat %}
"""

    prompt = RichPromptTemplate(prompt_str)

    messages = prompt.format_messages(name="John", your_image=image_url)
    assert len(messages) == 1
    assert len(messages[0].blocks) == 2
    assert messages[0].blocks[0].block_type == "text"
    assert messages[0].blocks[1].block_type == "image"
    assert str(messages[0].blocks[1].url) == image_url


def test_prompt_with_text_blocks():
    tb = TextBlock(text="hello world")

    prompt_str = """
{% chat role='user' %}
{{ text_block.text }}
{% endchat %}
"""

    prompt = RichPromptTemplate(prompt_str)

    messages = prompt.format_messages(text_block=tb)
    assert len(messages) == 1
    assert len(messages[0].blocks) == 1
    assert messages[0].blocks[0].block_type == "text"
    assert messages[0].blocks[0].text == "hello world"


def test_prompt_with_image_blocks(png_1px):
    ib = ImageBlock(image=png_1px)

    prompt_str = """
{% chat role='user' %}
{{ image.inline_url() | image }}
{% endchat %}
"""

    prompt = RichPromptTemplate(prompt_str)

    messages = prompt.format_messages(image=ib)
    assert len(messages) == 1
    assert len(messages[0].blocks) == 1
    assert messages[0].blocks[0].block_type == "image"
    assert messages[0].blocks[0].resolve_image().read() == ib.resolve_image().read()


def test_prompt_with_audio_blocks(mp3_bytes):
    ab = AudioBlock(audio=mp3_bytes)

    prompt_str = """
{% chat role='user' %}
{{ audio.inline_url() | audio }}
{% endchat %}
"""

    prompt = RichPromptTemplate(prompt_str)

    messages = prompt.format_messages(audio=ab)
    assert len(messages) == 1
    assert len(messages[0].blocks) == 1
    assert messages[0].blocks[0].block_type == "audio"
    assert messages[0].blocks[0].resolve_audio().read() == ab.resolve_audio().read()


def test_prompt_with_video_blocks(mp4_bytes):
    vb = VideoBlock(video=mp4_bytes)

    prompt_str = """
{% chat role='user' %}
{{ video.inline_url() | video }}
{% endchat %}
"""

    prompt = RichPromptTemplate(prompt_str)

    messages = prompt.format_messages(video=vb)
    assert len(messages) == 1
    assert len(messages[0].blocks) == 1
    assert messages[0].blocks[0].block_type == "video"
    assert messages[0].blocks[0].resolve_video().read() == vb.resolve_video().read()


def test_prompt_with_document_blocks(mock_pdf_bytes):
    db = DocumentBlock(data=mock_pdf_bytes)

    prompt_str = """
{% chat role='user' %}
{{ document.inline_url() | document }}
{% endchat %}
"""

    prompt = RichPromptTemplate(prompt_str)

    messages = prompt.format_messages(document=db)
    assert len(messages) == 1
    assert len(messages[0].blocks) == 1
    assert messages[0].blocks[0].block_type == "document"
    assert (
        messages[0].blocks[0].resolve_document().read() == db.resolve_document().read()
    )


@pytest.mark.parametrize("role", ["user", "assistant", "system"])
def test_prompt_with_chat_message(role, png_1px, mp3_bytes, mp4_bytes, mock_pdf_bytes):
    chat_message = ChatMessage(
        blocks=[
            TextBlock(text="hello world"),
            ImageBlock(image=png_1px),
            AudioBlock(audio=mp3_bytes),
            VideoBlock(video=mp4_bytes),
            DocumentBlock(data=mock_pdf_bytes),
        ],
        role=role,
    )
    message_block = """{% for block in message.blocks %}
{% if block.block_type == 'text' %}
{{ block.text }}
{% elif block.block_type == 'image' %}
{{ block.inline_url() | image }}
{% elif block.block_type == 'audio' %}
{{ block.inline_url() | audio }}
{% elif block.block_type == 'video' %}
{{ block.inline_url() | video }}
{% elif block.block_type == 'document' %}
{{ block.inline_url() | document }}
{% endif %}
{% endfor %}"""

    prompt_str = (
        """
{% if message.role.value == "user" %}
{% chat role="user" %}
"""
        + message_block
        + """
{% endchat %}
{% elif message.role.value == "assistant" %}
{% chat role="assistant" %}
"""
        + message_block
        + """
{% endchat %}
{% elif message.role.value == "system" %}
{% chat role="system" %}
"""
        + message_block
        + """
{% endchat %}
{% endif %}
"""
    )

    prompt = RichPromptTemplate(prompt_str)

    messages = prompt.format_messages(message=chat_message)
    assert len(messages) == 1
    assert messages[0].role == role
    assert len(messages[0].blocks) == 5


def test_prompt_with_chat_messages(png_1px, mp3_bytes, mp4_bytes, mock_pdf_bytes):
    blocks = [
        TextBlock(text="hello world"),
        ImageBlock(image=png_1px),
        AudioBlock(audio=mp3_bytes),
        VideoBlock(video=mp4_bytes),
        DocumentBlock(data=mock_pdf_bytes),
    ]
    chat_messages = [
        ChatMessage(blocks=blocks, role="user"),
        ChatMessage(blocks=blocks, role="assistant"),
        ChatMessage(blocks=blocks, role="system"),
    ]
    message_block = """{% for block in message.blocks %}
    {% if block.block_type == 'text' %}
    {{ block.text }}
    {% elif block.block_type == 'image' %}
    {{ block.inline_url() | image }}
    {% elif block.block_type == 'audio' %}
    {{ block.inline_url() | audio }}
    {% elif block.block_type == 'video' %}
    {{ block.inline_url() | video }}
    {% elif block.block_type == 'document' %}
    {{ block.inline_url() | document }}
    {% endif %}
    {% endfor %}"""

    prompt_str = (
        """
    {% for message in messages %}
    {% if message.role.value == "user" %}
    {% chat role="user" %}
    """
        + message_block
        + """
    {% endchat %}
    {% elif message.role.value == "assistant" %}
    {% chat role="assistant" %}
    """
        + message_block
        + """
    {% endchat %}
    {% elif message.role.value == "system" %}
    {% chat role="system" %}
    """
        + message_block
        + """
    {% endchat %}
    {% endif %}
    {% endfor %}
    """
    )

    prompt = RichPromptTemplate(prompt_str)

    messages = prompt.format_messages(messages=chat_messages)
    assert len(messages) == 3
    for i, role in enumerate(["user", "assistant", "system"]):
        assert messages[i].role == role
        assert len(messages[i].blocks) == 5
