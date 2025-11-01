import base64
from io import BytesIO
from pathlib import Path
from unittest import mock
import pytest
import httpx

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    ImageBlock,
    MessageRole,
    TextBlock,
    DocumentBlock,
    VideoBlock,
    AudioBlock,
    CachePoint,
    CacheControl,
    ThinkingBlock,
    ToolCallBlock,
)
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.bridge.pydantic import ValidationError
from llama_index.core.schema import ImageDocument
from pydantic import AnyUrl


@pytest.fixture()
def empty_bytes() -> bytes:
    return b""


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
def pdf_base64(mock_pdf_bytes) -> bytes:
    return base64.b64encode(mock_pdf_bytes)


@pytest.fixture()
def mp4_bytes() -> bytes:
    # Minimal fake MP4 header bytes (ftyp box)
    return b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"


@pytest.fixture()
def mp4_base64(mp4_bytes: bytes) -> bytes:
    return base64.b64encode(mp4_bytes)


def test_chat_message_from_str():
    m = ChatMessage.from_str(content="test content")
    assert m.content == "test content"
    assert len(m.blocks) == 1
    assert type(m.blocks[0]) is TextBlock
    assert m.blocks[0].text == "test content"


def test_chat_message_content_legacy_get():
    m = ChatMessage(content="test content")
    assert m.content == "test content"
    assert len(m.blocks) == 1
    assert type(m.blocks[0]) is TextBlock
    assert m.blocks[0].text == "test content"

    m = ChatMessage(role="user", content="test content")
    assert m.role == "user"
    assert m.content == "test content"
    assert len(m.blocks) == 1
    assert type(m.blocks[0]) is TextBlock
    assert m.blocks[0].text == "test content"

    m = ChatMessage(
        content=[TextBlock(text="test content 1"), TextBlock(text="test content 2")]
    )
    assert m.content == "test content 1\ntest content 2"
    assert len(m.blocks) == 2
    assert all(type(block) is TextBlock for block in m.blocks)


def test_chat_message_content_legacy_set():
    m = ChatMessage()
    m.content = "test content"
    assert len(m.blocks) == 1
    assert type(m.blocks[0]) is TextBlock
    assert m.blocks[0].text == "test content"

    m = ChatMessage(content="some original content")
    m.content = "test content"
    assert len(m.blocks) == 1
    assert type(m.blocks[0]) is TextBlock
    assert m.blocks[0].text == "test content"

    m = ChatMessage(content=[TextBlock(text="test content"), ImageBlock()])
    with pytest.raises(ValueError):
        m.content = "test content"


def test_chat_message_content_returns_empty_string():
    m = ChatMessage(content=[TextBlock(text="test content"), ImageBlock()])
    assert m.content == "test content"
    m = ChatMessage()
    assert m.content is None


def test_chat_message__str__():
    assert str(ChatMessage(content="test content")) == "user: test content"


def test_chat_message_serializer():
    class SimpleModel(BaseModel):
        some_field: str = ""

    m = ChatMessage(
        content="test content",
        additional_kwargs={"some_list": ["a", "b", "c"], "some_object": SimpleModel()},
    )
    assert m.model_dump() == {
        "role": MessageRole.USER,
        "additional_kwargs": {
            "some_list": ["a", "b", "c"],
            "some_object": {"some_field": ""},
        },
        "blocks": [{"block_type": "text", "text": "test content"}],
    }


def test_chat_message_legacy_roundtrip():
    legacy_message = {
        "role": MessageRole.USER,
        "content": "foo",
        "additional_kwargs": {},
    }
    m = ChatMessage(**legacy_message)
    assert m.model_dump() == {
        "additional_kwargs": {},
        "blocks": [{"block_type": "text", "text": "foo"}],
        "role": MessageRole.USER,
    }


def test_image_block_resolve_image(png_1px: bytes, png_1px_b64: bytes):
    b = ImageBlock(image=png_1px)

    img = b.resolve_image()
    assert isinstance(img, BytesIO)
    assert img.read() == png_1px

    img = b.resolve_image(as_base64=True)
    assert isinstance(img, BytesIO)
    assert img.read() == png_1px_b64


def test_image_block_resolve_image_buffer(png_1px: bytes):
    buffer = BytesIO(png_1px)
    b = ImageBlock(image=buffer)

    img = b.resolve_image()
    assert isinstance(img, BytesIO)
    assert img.read() == png_1px


def test_image_block_resolve_image_path(
    tmp_path: Path, png_1px_b64: bytes, png_1px: bytes
):
    png_path = tmp_path / "test.png"
    png_path.write_bytes(png_1px)

    b = ImageBlock(path=png_path)
    img = b.resolve_image()
    assert isinstance(img, BytesIO)
    assert img.read() == png_1px

    img = b.resolve_image(as_base64=True)
    assert isinstance(img, BytesIO)
    assert img.read() == png_1px_b64


def test_image_block_resolve_image_url(png_1px_b64: bytes, png_1px: bytes):
    with mock.patch("llama_index.core.utils.requests") as mocked_req:
        url_str = "http://example.com"
        mocked_req.get.return_value = mock.MagicMock(content=png_1px)
        b = ImageBlock(url=AnyUrl(url=url_str))
        img = b.resolve_image()
        assert isinstance(img, BytesIO)
        assert img.read() == png_1px

        img = b.resolve_image(as_base64=True)
        assert isinstance(img, BytesIO)
        assert img.read() == png_1px_b64


def test_image_block_resolve_image_data_url_base64(png_1px_b64: bytes, png_1px: bytes):
    # Test data URL with base64 encoding
    data_url = f"data:image/png;base64,{png_1px_b64.decode('utf-8')}"
    b = ImageBlock(url=AnyUrl(url=data_url))

    img = b.resolve_image()
    assert isinstance(img, BytesIO)
    assert img.read() == png_1px

    img = b.resolve_image(as_base64=True)
    assert isinstance(img, BytesIO)
    assert img.read() == png_1px_b64


def test_image_block_resolve_image_data_url_plain_text():
    # Test data URL with plain text (no base64)
    test_text = "Hello, World!"
    data_url = f"data:text/plain,{test_text}"
    b = ImageBlock(url=AnyUrl(url=data_url))

    img = b.resolve_image()
    assert isinstance(img, BytesIO)
    assert img.read() == test_text.encode("utf-8")

    img = b.resolve_image(as_base64=True)
    assert isinstance(img, BytesIO)
    assert img.read() == base64.b64encode(test_text.encode("utf-8"))


def test_image_block_resolve_image_data_url_invalid():
    # Test invalid data URL format (missing comma)
    invalid_data_url = "data:image/png;base64"
    b = ImageBlock(url=AnyUrl(url=invalid_data_url))

    with pytest.raises(
        ValueError, match="Invalid data URL format: missing comma separator"
    ):
        b.resolve_image()


def test_image_block_resolve_error():
    with pytest.raises(
        ValueError, match="No valid source provided to resolve binary data!"
    ):
        b = ImageBlock()
        b.resolve_image()


def test_image_block_store_as_anyurl():
    url_str = "http://example.com"
    b = ImageBlock(url=url_str)
    assert b.url == AnyUrl(url=url_str)


def test_image_block_store_as_base64(png_1px_b64: bytes, png_1px: bytes):
    # Store regular bytes
    assert ImageBlock(image=png_1px).image == png_1px_b64
    # Store already encoded data
    assert ImageBlock(image=png_1px_b64).image == png_1px_b64


def test_legacy_image_additional_kwargs(png_1px_b64: bytes):
    image_doc = ImageDocument(image=png_1px_b64)
    msg = ChatMessage(additional_kwargs={"images": [image_doc]})
    assert len(msg.blocks) == 1
    assert msg.blocks[0].image == png_1px_b64


def test_chat_response():
    message = ChatMessage("some content")
    cr = ChatResponse(message=message)
    assert str(cr) == str(message)


def test_completion_response():
    cr = CompletionResponse(text="some text")
    assert str(cr) == "some text"


def test_document_block_from_bytes(mock_pdf_bytes: bytes, pdf_base64: bytes):
    document = DocumentBlock(data=mock_pdf_bytes, document_mimetype="application/pdf")
    assert document.title == "input_document"
    assert document.document_mimetype == "application/pdf"
    assert pdf_base64 == document.data


def test_document_block_from_b64(pdf_base64: bytes):
    document = DocumentBlock(data=pdf_base64)
    assert document.title == "input_document"
    assert pdf_base64 == document.data


def test_document_block_from_path(tmp_path: Path, pdf_url: str):
    pdf_path = tmp_path / "test.pdf"
    pdf_content = httpx.get(pdf_url).content
    pdf_path.write_bytes(pdf_content)
    document = DocumentBlock(path=pdf_path.__str__())
    file_buffer = document.resolve_document()
    assert isinstance(file_buffer, BytesIO)
    file_bytes = file_buffer.read()
    document._guess_mimetype()
    assert document.document_mimetype == "application/pdf"
    fm = document.guess_format()
    assert fm == "pdf"
    b64_string = document._get_b64_string(file_buffer)
    try:
        base64.b64decode(b64_string, validate=True)
        string_base64_encoded = True
    except Exception:
        string_base64_encoded = False
    assert string_base64_encoded
    b64_bytes = document._get_b64_bytes(file_buffer)
    try:
        base64.b64decode(b64_bytes, validate=True)
        bytes_base64_encoded = True
    except Exception:
        bytes_base64_encoded = False
    assert bytes_base64_encoded
    assert document.title == "input_document"


def test_document_block_from_url(pdf_url: str):
    document = DocumentBlock(url=pdf_url, title="dummy_pdf")
    file_buffer = document.resolve_document()
    assert isinstance(file_buffer, BytesIO)
    file_bytes = file_buffer.read()
    document._guess_mimetype()
    assert document.document_mimetype == "application/pdf"
    fm = document.guess_format()
    assert fm == "pdf"
    b64_string = document._get_b64_string(file_buffer)
    try:
        base64.b64decode(b64_string, validate=True)
        string_base64_encoded = True
    except Exception as e:
        string_base64_encoded = False
    assert string_base64_encoded
    b64_bytes = document._get_b64_bytes(file_buffer)
    try:
        base64.b64decode(b64_bytes, validate=True)
        bytes_base64_encoded = True
    except Exception:
        bytes_base64_encoded = False
    assert bytes_base64_encoded
    assert document.title == "dummy_pdf"


def test_empty_bytes(empty_bytes: bytes, png_1px: bytes):
    errors = []
    try:
        DocumentBlock(data=empty_bytes).resolve_document()
        errors.append(0)
    except ValueError:
        errors.append(1)
    try:
        AudioBlock(audio=empty_bytes).resolve_audio()
        errors.append(0)
    except ValueError:
        errors.append(1)
    try:
        ImageBlock(image=empty_bytes).resolve_image()
        errors.append(0)
    except ValueError:
        errors.append(1)
    try:
        ImageBlock(image=png_1px).resolve_image()
        errors.append(0)
    except ValueError:
        errors.append(1)
    assert sum(errors) == 3


def test_cache_control() -> None:
    cp = CachePoint(cache_control=CacheControl(type="ephemeral"))
    assert isinstance(cp.model_dump()["cache_control"], dict)
    assert cp.model_dump()["cache_control"]["type"] == "ephemeral"
    with pytest.raises(ValidationError):
        CachePoint.model_validate({"cache_control": "default"})


def test_video_block_resolve_video_bytes(mp4_bytes: bytes, mp4_base64: bytes):
    b = VideoBlock(video=mp4_bytes)

    vid = b.resolve_video()
    assert isinstance(vid, BytesIO)
    assert vid.read() == mp4_bytes

    vid = b.resolve_video(as_base64=True)
    assert isinstance(vid, BytesIO)
    assert vid.read() == mp4_base64


def test_video_block_resolve_video_buffer(mp4_bytes: bytes):
    buffer = BytesIO(mp4_bytes)
    b = VideoBlock(video=buffer)

    vid = b.resolve_video()
    assert isinstance(vid, BytesIO)
    assert vid.read() == mp4_bytes


def test_video_block_resolve_video_path(
    tmp_path: Path, mp4_bytes: bytes, mp4_base64: bytes
):
    mp4_path = tmp_path / "test.mp4"
    mp4_path.write_bytes(mp4_bytes)

    b = VideoBlock(path=mp4_path)
    vid = b.resolve_video()
    assert isinstance(vid, BytesIO)
    assert vid.read() == mp4_bytes

    vid = b.resolve_video(as_base64=True)
    assert isinstance(vid, BytesIO)
    assert vid.read() == mp4_base64


def test_video_block_resolve_video_url(mp4_bytes: bytes, mp4_base64: bytes):
    with mock.patch("llama_index.core.utils.requests") as mocked_req:
        url_str = "http://example.com/video.mp4"
        mocked_req.get.return_value = mock.MagicMock(content=mp4_bytes)
        b = VideoBlock(url=AnyUrl(url=url_str))
        vid = b.resolve_video()
        assert isinstance(vid, BytesIO)
        assert vid.read() == mp4_bytes

        vid = b.resolve_video(as_base64=True)
        assert isinstance(vid, BytesIO)
        assert vid.read() == mp4_base64


def test_video_block_resolve_video_data_url_base64(mp4_bytes: bytes, mp4_base64: bytes):
    # Test data URL with base64 encoding
    data_url = f"data:video/mp4;base64,{mp4_base64.decode('utf-8')}"
    b = VideoBlock(url=AnyUrl(url=data_url))

    vid = b.resolve_video()
    assert isinstance(vid, BytesIO)
    assert vid.read() == mp4_bytes

    vid = b.resolve_video(as_base64=True)
    assert isinstance(vid, BytesIO)
    assert vid.read() == mp4_base64


def test_video_block_resolve_error():
    b = VideoBlock()
    with pytest.raises(ValueError, match="No valid source provided"):
        b.resolve_video()


def test_video_block_store_as_anyurl():
    url_str = "http://example.com/video.mp4"
    b = VideoBlock(url=url_str)
    assert isinstance(b.url, AnyUrl)
    assert str(b.url) == url_str


def test_video_block_store_as_base64(mp4_bytes: bytes, mp4_base64: bytes):
    # Store regular bytes
    assert VideoBlock(video=mp4_bytes).video == mp4_base64
    # Store already encoded data
    assert VideoBlock(video=mp4_base64).video == mp4_base64


def test_thinking_block():
    block = ThinkingBlock()
    assert block.block_type == "thinking"
    assert block.additional_information == {}
    assert block.content is None
    assert block.num_tokens is None
    block = ThinkingBlock(
        content="hello world",
        num_tokens=100,
        additional_information={"total_thinking_tokens": 1000},
    )
    assert block.block_type == "thinking"
    assert block.additional_information == {"total_thinking_tokens": 1000}
    assert block.content == "hello world"
    assert block.num_tokens == 100


def test_tool_call_block():
    default_block = ToolCallBlock(tool_name="hello_world")
    assert default_block.block_type == "tool_call"
    assert default_block.tool_call_id is None
    assert default_block.tool_name == "hello_world"
    assert default_block.tool_kwargs == {}
    custom_block = ToolCallBlock(
        tool_name="hello_world",
        tool_call_id="1",
        tool_kwargs={"test": 1},
    )
    assert custom_block.tool_call_id == "1"
    assert custom_block.tool_name == "hello_world"
    assert custom_block.tool_kwargs == {"test": 1}
