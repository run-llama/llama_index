import base64
from io import BytesIO
from pathlib import Path
from unittest import mock
from unittest.mock import Mock

import pytest
import httpx
from tinytag import UnsupportedFormatError, TinyTag

from llama_index.core import get_tokenizer
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
from llama_index.core.node_parser import TokenTextSplitter
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
def mp3_bytes() -> bytes:
    """
    Small mp3 file bytes (0.2 seconds of audio).
    """
    return b"ID3\x04\x00\x00\x00\x00\x01\tTXXX\x00\x00\x00\x12\x00\x00\x03major_brand\x00isom\x00TXXX\x00\x00\x00\x13\x00\x00\x03minor_version\x00512\x00TXXX\x00\x00\x00$\x00\x00\x03compatible_brands\x00isomiso2avc1mp41\x00TSSE\x00\x00\x00\x0e\x00\x00\x03Lavf62.3.100\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xf3X\xc0\x00\x00\x00\x00\x00\x00\x00\x00\x00Info\x00\x00\x00\x0f\x00\x00\x00\x06\x00\x00\x03<\x00YYYYYYYYYYYYYYYYzzzzzzzzzzzzzzzzz\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\x9b\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xbd\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xde\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\x00\x00\x00\x00Lavf\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00$\x00\x00\x00\x00\x00\x00\x00\x00\x03<\xa6\xbc`\x8e\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xf38\xc4\x00\x00\x00\x03H\x00\x00\x00\x00LAME3.100UUUUUUUUUUUUUUUUUUUUULAME3.100UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\xff\xf38\xc4_\x00\x00\x03H\x00\x00\x00\x00UUUUUUUUUUUUUUUUUUUUUUUUUUUUUULAME3.100UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\xff\xf38\xc4\xa0\x00\x00\x03H\x00\x00\x00\x00UUUUUUUUUUUUUUUUUUUUUUUUUUUUUULAME3.100UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\xff\xf38\xc4\xa0\x00\x00\x03H\x00\x00\x00\x00UUUUUUUUUUUUUUUUUUUUUUUUUUUUUULAME3.100UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\xff\xf38\xc4\xa0\x00\x00\x03H\x00\x00\x00\x00UUUUUUUUUUUUUUUUUUUUUUUUUUUUUULAME3.100UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\xff\xf38\xc4\xa0\x00\x00\x03H\x00\x00\x00\x00UUUUUUUUUUUUUUUUUUUUUUUUUUUUUULAME3.100UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU"


@pytest.fixture()
def mp3_base64(mp3_bytes: bytes) -> bytes:
    return base64.b64encode(mp3_bytes)


@pytest.fixture()
def mp4_bytes() -> bytes:
    # Minimal fake MP4 header bytes (ftyp box)
    return b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"


@pytest.fixture()
def mp4_base64(mp4_bytes: bytes) -> bytes:
    return base64.b64encode(mp4_bytes)


@pytest.fixture()
def mock_tiny_tag_error():
    def raise_tiny_tag_error(*args, **kwargs):
        raise UnsupportedFormatError

    with mock.patch.object(
        TinyTag, "get", side_effect=raise_tiny_tag_error
    ) as mock_tinytag:
        yield mock_tinytag


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


def test_chat_response():
    message = ChatMessage("some content")
    cr = ChatResponse(message=message)
    assert str(cr) == str(message)


def test_completion_response():
    cr = CompletionResponse(text="some text")
    assert str(cr) == "some text"


@pytest.mark.asyncio
async def test_text_block_aestimate_tokens_default_tokenizer():
    tb = TextBlock(text="Hello world!")

    tknzr = get_tokenizer()
    assert await tb.aestimate_tokens() == len(tknzr(tb.text))


@pytest.mark.asyncio
async def test_text_block_aestimate_tokens_custom_tokenizer():
    tb = TextBlock(text="Hello world!")

    mock_tknzer = Mock(spec=type(get_tokenizer()))
    mock_tknzer.return_value = list(range(100))
    assert await tb.aestimate_tokens(tokenizer=mock_tknzer) == 100


@pytest.mark.asyncio
async def test_text_block_asplit_no_overlap():
    tb = TextBlock(text="Hello world! This is a test.")

    chunks = await tb.asplit(max_tokens=3)
    splitter = TokenTextSplitter(chunk_size=3, chunk_overlap=0)
    assert len(chunks) == len(splitter.split_text(tb.text))


@pytest.mark.asyncio
async def test_text_block_atruncate():
    tb = TextBlock(text="Hello world! This is a test.")
    truncated_tb = await tb.atruncate(max_tokens=4)
    truncated_tb_reverse = await tb.atruncate(max_tokens=4, reverse=True)
    assert await tb.aestimate_tokens() > 4
    assert await truncated_tb.aestimate_tokens() <= 4
    assert await truncated_tb_reverse.aestimate_tokens() <= 4
    assert truncated_tb.text == "Hello world! This"
    assert truncated_tb_reverse.text == "is a test."


@pytest.mark.asyncio
async def test_text_block_amerge():
    tb1 = TextBlock(text="Hello world!")
    tb2 = TextBlock(text="This is a test.")
    merged_tb = await TextBlock.amerge([tb1, tb2], chunk_size=100)
    assert len(merged_tb) == 1
    assert merged_tb[0].text == "Hello world! This is a test."


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


@pytest.mark.asyncio
async def test_image_block_aestimate_tokens(png_1px_b64: bytes):
    ib = ImageBlock(image=png_1px_b64)
    assert await ib.aestimate_tokens() == 2125


@pytest.mark.asyncio
async def test_image_block_asplit(png_1px_b64: bytes):
    ib = ImageBlock(image=png_1px_b64)
    chunks = await ib.asplit(max_tokens=2)

    # Images are not splittable
    assert chunks == [ib]


@pytest.mark.asyncio
async def test_image_block_atruncate(png_1px_b64: bytes):
    ib = ImageBlock(image=png_1px_b64)
    truncated_ib = await ib.atruncate(max_tokens=2)
    truncated_ib_reverse = await ib.atruncate(max_tokens=2, reverse=True)
    # Images are not truncatable
    assert truncated_ib == ib
    assert truncated_ib_reverse == ib


@pytest.mark.asyncio
async def test_image_block_amerge(png_1px_b64: bytes):
    ib1 = ImageBlock(image=png_1px_b64)
    ib2 = ImageBlock(image=png_1px_b64)
    merged = await ImageBlock.amerge([ib1, ib2], chunk_size=1000)

    # Images are not mergeable
    assert merged == [ib1, ib2]


@pytest.mark.asyncio
async def test_audio_block_aestimate_tokens(mp3_bytes: bytes):
    ab = AudioBlock(audio=mp3_bytes)
    assert await ab.aestimate_tokens() == 32  # based on 1 token per 4 bytes


@pytest.mark.asyncio
async def test_audio_block_aestimate_tokens_tiny_tag_error(
    mp3_base64: bytes, mock_tiny_tag_error
):
    """TinyTag is able to read mp3 metadata without ffmpeg installed."""
    ab = AudioBlock(audio=mp3_base64)
    assert await ab.aestimate_tokens() == 256  # Fallback


@pytest.mark.asyncio
async def test_audio_block_asplit(mp3_bytes: bytes, mp3_base64: bytes):
    ab = AudioBlock(audio=mp3_bytes)
    chunks = await ab.asplit(max_tokens=2)

    # No splitting occurs
    assert chunks == [ab]


@pytest.mark.asyncio
async def test_audio_block_atruncate(mp3_bytes: bytes, mp3_base64: bytes):
    ab = AudioBlock(audio=mp3_bytes)
    truncated_ab = await ab.atruncate(max_tokens=16)
    truncated_ab_reverse = await ab.atruncate(max_tokens=16, reverse=True)
    # No truncation occurs
    assert truncated_ab == ab
    assert truncated_ab_reverse == ab


@pytest.mark.asyncio
async def test_audio_block_amerge(mp3_bytes: bytes):
    ab1 = AudioBlock(audio=mp3_bytes)
    ab2 = AudioBlock(audio=mp3_bytes)
    merged_abs = await AudioBlock.amerge([ab1, ab2], chunk_size=1000)

    # No merging occurs
    assert len(merged_abs) == 2
    assert merged_abs == [ab1, ab2]


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


@pytest.mark.asyncio
async def test_video_block_aestimate_tokens(mp4_base64: bytes):
    """TinyTag fails for most video types, including this mp4 type."""
    vb = VideoBlock(video=mp4_base64)
    assert await vb.aestimate_tokens() == 256 * 8  # Fallback


@pytest.mark.asyncio
async def test_video_block_asplit(mp4_bytes: bytes, mp4_base64: bytes):
    vb = VideoBlock(video=mp4_bytes)
    chunks = await vb.asplit(max_tokens=500)

    # No splitting occurs
    assert chunks == [vb]


@pytest.mark.asyncio
async def test_video_block_atruncate(mp4_bytes: bytes, mp4_base64: bytes):
    vb = VideoBlock(video=mp4_bytes)
    truncated_vb = await vb.atruncate(max_tokens=500)
    truncated_vb_reverse = await vb.atruncate(max_tokens=500, reverse=True)
    # No truncation occurs
    assert truncated_vb == vb
    assert truncated_vb_reverse == vb


@pytest.mark.asyncio
async def test_video_block_amerge(mp4_bytes: bytes):
    vb1 = VideoBlock(video=mp4_bytes)
    vb2 = VideoBlock(video=mp4_bytes)
    merged_vbs = await VideoBlock.amerge([vb1, vb2], chunk_size=2000)

    # No merging occurs
    assert len(merged_vbs) == 2
    assert merged_vbs == [vb1, vb2]


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


def test_document_block_empty_bytes(empty_bytes: bytes, png_1px: bytes):
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


@pytest.mark.asyncio
async def test_document_block_aestimate_tokens(mock_pdf_bytes: bytes):
    document = DocumentBlock(data=mock_pdf_bytes, document_mimetype="application/pdf")
    # Fallback: we currently don't estimate tokens for documents since it's too complicated to handle
    # all the different document types. Essentially kicking the can here.
    assert await document.aestimate_tokens() == 512


@pytest.mark.asyncio
async def test_document_block_asplit(mock_pdf_bytes: bytes):
    document = DocumentBlock(data=mock_pdf_bytes, document_mimetype="application/pdf")
    chunks = await document.asplit(max_tokens=100)
    # We dont split documents currently
    assert chunks == [document]


@pytest.mark.asyncio
async def test_document_block_atruncate(mock_pdf_bytes: bytes):
    document = DocumentBlock(data=mock_pdf_bytes, document_mimetype="application/pdf")
    truncated_document = await document.atruncate(max_tokens=100)
    truncated_document_reverse = await truncated_document.atruncate(
        max_tokens=100, reverse=True
    )
    # We dont truncate documents currently
    assert truncated_document == document
    assert truncated_document_reverse == document


@pytest.mark.asyncio
async def test_document_block_amerge(mock_pdf_bytes: bytes):
    document1 = DocumentBlock(data=mock_pdf_bytes, document_mimetype="application/pdf")
    document2 = DocumentBlock(data=mock_pdf_bytes, document_mimetype="application/pdf")
    merged = await DocumentBlock.amerge([document1, document2], chunk_size=1000)
    # We dont merge documents currently
    assert merged == [document1, document2]


def test_cache_control() -> None:
    cp = CachePoint(cache_control=CacheControl(type="ephemeral"))
    assert isinstance(cp.model_dump()["cache_control"], dict)
    assert cp.model_dump()["cache_control"]["type"] == "ephemeral"
    with pytest.raises(ValidationError):
        CachePoint.model_validate({"cache_control": "default"})


@pytest.mark.asyncio
async def test_cache_control_aestimate_tokens():
    cp = CachePoint(cache_control=CacheControl(type="ephemeral"))
    # No content length for ephemeral cache control
    assert await cp.aestimate_tokens() == 0


@pytest.mark.asyncio
async def test_cache_control_asplit():
    cp = CachePoint(cache_control=CacheControl(type="ephemeral"))
    chunks = await cp.asplit(max_tokens=10)
    # Cache control points are not splittable
    assert chunks == [cp]


@pytest.mark.asyncio
async def test_cache_control_atruncate():
    cp = CachePoint(cache_control=CacheControl(type="ephemeral"))
    truncated_cp = await cp.atruncate(max_tokens=10)
    truncated_cp_reverse = await cp.atruncate(max_tokens=10, reverse=True)
    # Cache control points are not truncatable
    assert truncated_cp == cp
    assert truncated_cp_reverse == cp


@pytest.mark.asyncio
async def test_cache_control_amerge():
    cp1 = CachePoint(cache_control=CacheControl(type="ephemeral"))
    cp2 = CachePoint(cache_control=CacheControl(type="ephemeral"))
    merged = await CachePoint.amerge([cp1, cp2], chunk_size=100)
    # Cache control points are not mergeable
    assert merged == [cp1, cp2]


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


@pytest.mark.asyncio
async def test_thinking_block_aestimate_tokens():
    block1 = ThinkingBlock(content="Some Content", num_tokens=150)
    block2 = ThinkingBlock(content="Some Content")

    assert await block1.aestimate_tokens() == block1.num_tokens
    assert (
        await block2.aestimate_tokens()
        == await TextBlock(text=block2.content).aestimate_tokens()
    )


@pytest.mark.asyncio
async def test_thinking_block_asplit():
    block = ThinkingBlock(content="This is a test of the ThinkingBlock split method.")
    chunks = await block.asplit(max_tokens=5)

    # Thinking blocks are not splittable
    assert chunks == [block]


@pytest.mark.asyncio
async def test_thinking_block_atruncate():
    block = ThinkingBlock(
        content="This is a test of the ThinkingBlock truncate method."
    )
    truncated_block = await block.atruncate(max_tokens=5)
    truncated_block_reverse = await block.atruncate(max_tokens=5, reverse=True)
    # Thinking blocks are not truncatable
    assert truncated_block == block
    assert truncated_block_reverse == block


@pytest.mark.asyncio
async def test_thinking_block_amerge():
    block1 = ThinkingBlock(content="This is the first ThinkingBlock.")
    block2 = ThinkingBlock(content="This is the second ThinkingBlock.")
    merged = await ThinkingBlock.amerge([block1, block2], chunk_size=100)

    # Thinking blocks are not mergeable
    assert merged == [block1, block2]


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


@pytest.mark.asyncio
async def test_tool_call_block_aestimate_tokens():
    block = ToolCallBlock(
        tool_name="example_tool", tool_kwargs={"param1": "value1", "param2": 42}
    )
    assert (
        await block.aestimate_tokens()
        == await TextBlock(text=block.model_dump_json()).aestimate_tokens()
    )


@pytest.mark.asyncio
async def test_tool_call_block_asplit():
    block = ToolCallBlock(
        tool_name="example_tool", tool_kwargs={"param1": "value1", "param2": 42}
    )

    # ToolCallBlocks are not splittable
    assert await block.asplit(max_tokens=5) == [block]


@pytest.mark.asyncio
async def test_tool_call_block_atruncate():
    block = ToolCallBlock(
        tool_name="example_tool", tool_kwargs={"param1": "value1", "param2": 42}
    )
    truncated_block = await block.atruncate(max_tokens=5)
    truncated_block_reverse = await block.atruncate(max_tokens=5, reverse=True)
    # ToolCallBlocks are not truncatable
    assert truncated_block == block
    assert truncated_block_reverse == block


@pytest.mark.asyncio
async def test_tool_call_block_amerge():
    block1 = ToolCallBlock(tool_name="example_tool_1", tool_kwargs={"param": "value1"})
    block2 = ToolCallBlock(tool_name="example_tool_2", tool_kwargs={"param": "value2"})
    merged = await ToolCallBlock.amerge([block1, block2], chunk_size=100)

    # ToolCallBlocks are not mergeable
    assert merged == [block1, block2]
