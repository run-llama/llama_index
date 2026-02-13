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
    CitableBlock,
    CitationBlock,
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


@pytest.mark.asyncio
async def test_chat_message_aestimate_tokens(
    png_1px, mp3_bytes, mp4_bytes, mock_pdf_bytes
):
    m = ChatMessage(
        blocks=[
            TextBlock(text="Hello world! This is a test."),
            ImageBlock(image=png_1px),
            AudioBlock(audio=mp3_bytes),
            VideoBlock(video=mp4_bytes),
            DocumentBlock(data=mock_pdf_bytes),
            CachePoint(cache_control=CacheControl(type="ephemeral")),
            CitableBlock(
                title="Test Title",
                source="Test Source",
                content=[
                    TextBlock(text="Citable block content."),
                    ImageBlock(image=png_1px),
                    DocumentBlock(data=mock_pdf_bytes),
                ],
            ),
            CitationBlock(
                title="Text Title",
                source="Text Source",
                cited_content=TextBlock(text="Citation block content."),
                additional_location_info={},
            ),
            CitationBlock(
                title="Image Title",
                source="Image Source",
                cited_content=ImageBlock(image=png_1px),
                additional_location_info={},
            ),
            ThinkingBlock(
                content="Thinking block content.",
            ),
            ThinkingBlock(num_tokens=50),
            ToolCallBlock(
                tool_call_id="tool_123",
                tool_name="Test Tool",
                tool_kwargs={"foo": "bar"},
            ),
        ]
    )

    assert await m.aestimate_tokens() == sum(
        [await block.aestimate_tokens() for block in m.blocks]
    )


@pytest.mark.asyncio
async def test_chat_message_asplit_non_recursive_types(
    png_1px, mp3_bytes, mp4_bytes, mock_pdf_bytes
):
    chat_message = ChatMessage(
        blocks=[
            TextBlock(text="Hello world! This is a test."),
            ImageBlock(image=png_1px),
            AudioBlock(audio=mp3_bytes),
            VideoBlock(video=mp4_bytes),
            DocumentBlock(data=mock_pdf_bytes),
            CachePoint(cache_control=CacheControl(type="ephemeral")),
            ThinkingBlock(
                content="Thinking block content.",
            ),
            ThinkingBlock(num_tokens=50),
            ToolCallBlock(
                tool_call_id="tool_123",
                tool_name="Test Tool",
                tool_kwargs={"foo": "bar"},
            ),
        ]
    )
    chunks = await chat_message.asplit(max_tokens=3)
    assert chunks == [
        ChatMessage(blocks=[chunk])
        for block in chat_message.blocks
        for chunk in await block.asplit(max_tokens=3)
    ]
    # TextBlock Should be split int 3 chunks
    assert sum([1 for chunk in chunks if isinstance(chunk.blocks[0], TextBlock)]) == 3
    # Image block should not be split
    assert sum([1 for chunk in chunks if isinstance(chunk.blocks[0], ImageBlock)]) == 1
    # Audio block should not be split
    assert sum([1 for chunk in chunks if isinstance(chunk.blocks[0], AudioBlock)]) == 1
    # Video block should not be split
    assert sum([1 for chunk in chunks if isinstance(chunk.blocks[0], VideoBlock)]) == 1
    # Document block should not be split
    assert (
        sum([1 for chunk in chunks if isinstance(chunk.blocks[0], DocumentBlock)]) == 1
    )
    # CachePoint block should not be split
    assert sum([1 for chunk in chunks if isinstance(chunk.blocks[0], CachePoint)]) == 1
    # ThinkingBlocks should not be split
    assert (
        sum([1 for chunk in chunks if isinstance(chunk.blocks[0], ThinkingBlock)]) == 2
    )
    # ToolCallBlock block should not be split
    assert (
        sum([1 for chunk in chunks if isinstance(chunk.blocks[0], ToolCallBlock)]) == 1
    )


@pytest.mark.asyncio
async def test_chat_message_asplit_recursive_types(png_1px, mock_pdf_bytes):
    chat_message = ChatMessage(
        blocks=[
            CitableBlock(
                title="Test Title",
                source="Test Source",
                content=[
                    TextBlock(text="Citable block content."),
                    ImageBlock(image=png_1px),
                    DocumentBlock(data=mock_pdf_bytes),
                ],
            ),
            CitationBlock(
                title="Text Title",
                source="Text Source",
                cited_content=TextBlock(text="Citation block content."),
                additional_location_info={},
            ),
            CitationBlock(
                title="Image Title",
                source="Image Source",
                cited_content=ImageBlock(image=png_1px),
                additional_location_info={},
            ),
        ]
    )
    chunks = await chat_message.asplit(max_tokens=3)

    assert chunks == [
        ChatMessage(blocks=[chunk])
        for block in chat_message.blocks
        for chunk in await block.asplit(max_tokens=3)
    ]

    # CitableBlock should be split into 4 chunks (2 text, 1 image, 1 document)
    assert (
        sum([1 for chunk in chunks if isinstance(chunk.blocks[0], CitableBlock)]) == 4
    )
    assert (
        sum(
            [
                1
                for chunk in chunks
                for rec_chunk in chunk.blocks[0].nested_blocks
                if isinstance(rec_chunk, TextBlock)
                and isinstance(chunk.blocks[0], CitableBlock)
            ]
        )
        == 2
    )
    assert (
        sum(
            [
                1
                for chunk in chunks
                for rec_chunk in chunk.blocks[0].nested_blocks
                if isinstance(rec_chunk, ImageBlock)
                and isinstance(chunk.blocks[0], CitableBlock)
            ]
        )
        == 1
    )
    assert (
        sum(
            [
                1
                for chunk in chunks
                for rec_chunk in chunk.blocks[0].nested_blocks
                if isinstance(rec_chunk, DocumentBlock)
                and isinstance(chunk.blocks[0], CitableBlock)
            ]
        )
        == 1
    )

    # CitationBlock with TextBlock should be split into 2 chunks
    # CitationBlock with ImageBlock should not be split (1 chunk)
    assert (
        sum([1 for chunk in chunks if isinstance(chunk.blocks[0], CitationBlock)]) == 3
    )
    assert (
        sum(
            [
                1
                for chunk in chunks
                for rec_chunk in chunk.blocks[0].nested_blocks
                if isinstance(rec_chunk, TextBlock)
                and isinstance(chunk.blocks[0], CitationBlock)
            ]
        )
        == 2
    )
    assert (
        sum(
            [
                1
                for chunk in chunks
                for rec_chunk in chunk.blocks[0].nested_blocks
                if isinstance(rec_chunk, ImageBlock)
                and isinstance(chunk.blocks[0], CitationBlock)
            ]
        )
        == 1
    )


@pytest.mark.asyncio
async def test_chat_message_atruncate_simple(
    png_1px, mp3_bytes, mp4_bytes, mock_pdf_bytes
):
    m1 = ChatMessage(blocks=[TextBlock(text="Hello world! This is a test.")])
    m2 = ChatMessage(blocks=[ImageBlock(image=png_1px)])
    m3 = ChatMessage(blocks=[AudioBlock(audio=mp3_bytes)])
    m4 = ChatMessage(blocks=[VideoBlock(video=mp4_bytes)])
    m5 = ChatMessage(blocks=[DocumentBlock(data=mock_pdf_bytes)])

    assert await m1.atruncate(max_tokens=3) == ChatMessage(
        blocks=[await m1.blocks[0].atruncate(max_tokens=3)]
    )
    assert await m1.atruncate(max_tokens=3, reverse=True) == ChatMessage(
        blocks=[await m1.blocks[0].atruncate(max_tokens=3, reverse=True)]
    )
    assert await m2.atruncate(max_tokens=3) == ChatMessage(
        blocks=[await m2.blocks[0].atruncate(max_tokens=3)]
    )
    assert await m2.atruncate(max_tokens=3, reverse=True) == ChatMessage(
        blocks=[await m2.blocks[0].atruncate(max_tokens=3, reverse=True)]
    )
    assert await m3.atruncate(max_tokens=3) == ChatMessage(
        blocks=[await m3.blocks[0].atruncate(max_tokens=3)]
    )
    assert await m3.atruncate(max_tokens=3, reverse=True) == ChatMessage(
        blocks=[await m3.blocks[0].atruncate(max_tokens=3, reverse=True)]
    )
    assert await m4.atruncate(max_tokens=3) == ChatMessage(
        blocks=[await m4.blocks[0].atruncate(max_tokens=3)]
    )
    assert await m4.atruncate(max_tokens=3, reverse=True) == ChatMessage(
        blocks=[await m4.blocks[0].atruncate(max_tokens=3, reverse=True)]
    )
    assert await m5.atruncate(max_tokens=3) == ChatMessage(
        blocks=[await m5.blocks[0].atruncate(max_tokens=3)]
    )
    assert await m5.atruncate(max_tokens=3, reverse=True) == ChatMessage(
        blocks=[await m5.blocks[0].atruncate(max_tokens=3, reverse=True)]
    )


@pytest.mark.asyncio
async def test_chat_message_atruncate_multiple_multimodal_blocks(
    png_1px, mp3_bytes, mp4_bytes, mock_pdf_bytes
):
    tb = TextBlock(text="Hello world! This is a test.")
    ib = ImageBlock(image=png_1px)
    ab = AudioBlock(audio=mp3_bytes)
    vb = VideoBlock(video=mp4_bytes)
    db = DocumentBlock(data=mock_pdf_bytes)

    chat_message = ChatMessage(blocks=[tb, ib, ab, vb, db])

    assert await chat_message.atruncate(max_tokens=3) == ChatMessage(
        blocks=[await chat_message.blocks[0].atruncate(max_tokens=3)]
    )
    assert await chat_message.atruncate(
        max_tokens=await tb.aestimate_tokens()
    ) == ChatMessage(blocks=[tb])
    assert await chat_message.atruncate(
        max_tokens=await tb.aestimate_tokens() + await ib.aestimate_tokens()
    ) == ChatMessage(blocks=[tb, ib])
    assert await chat_message.atruncate(
        max_tokens=sum([await b.aestimate_tokens() for b in [tb, ib, ab]])
    ) == ChatMessage(blocks=[tb, ib, ab])
    assert await chat_message.atruncate(
        max_tokens=sum([await b.aestimate_tokens() for b in [tb, ib, ab, vb]])
    ) == ChatMessage(blocks=[tb, ib, ab, vb])
    assert await chat_message.atruncate(
        max_tokens=await chat_message.aestimate_tokens()
    ) == ChatMessage(blocks=[tb, ib, ab, vb, db])

    # reverse truncation
    assert await chat_message.atruncate(
        max_tokens=await db.aestimate_tokens(), reverse=True
    ) == ChatMessage(blocks=[db])
    assert await chat_message.atruncate(
        max_tokens=await db.aestimate_tokens() + await vb.aestimate_tokens(),
        reverse=True,
    ) == ChatMessage(blocks=[vb, db])
    assert await chat_message.atruncate(
        max_tokens=sum([await b.aestimate_tokens() for b in [db, vb, ab]]),
        reverse=True,
    ) == ChatMessage(blocks=[ab, vb, db])
    assert await chat_message.atruncate(
        max_tokens=sum([await b.aestimate_tokens() for b in [db, vb, ab, ib]]),
        reverse=True,
    ) == ChatMessage(blocks=[ib, ab, vb, db])
    assert await chat_message.atruncate(
        max_tokens=3 + sum([await b.aestimate_tokens() for b in [db, vb, ab, ib]]),
        reverse=True,
    ) == ChatMessage(
        blocks=[await tb.atruncate(max_tokens=3, reverse=True), ib, ab, vb, db]
    )
    assert await chat_message.atruncate(
        max_tokens=await chat_message.aestimate_tokens(), reverse=True
    ) == ChatMessage(blocks=[tb, ib, ab, vb, db])


@pytest.mark.asyncio
async def test_chat_message_atruncate_recursive(png_1px, mock_pdf_bytes):
    tb = TextBlock(text="Block content")
    ib = ImageBlock(image=png_1px)
    db = DocumentBlock(data=mock_pdf_bytes)
    citable_block = CitableBlock(
        title="Test Title", source="Test Source", content=[tb, ib, db]
    )
    citation_block_text = CitationBlock(
        title="Text Title",
        source="Text Source",
        cited_content=tb,
        additional_location_info={},
    )
    citation_block_image = CitationBlock(
        title="Image Title",
        source="Image Source",
        cited_content=ib,
        additional_location_info={},
    )

    chat_message = ChatMessage(
        blocks=[citable_block, citation_block_text, citation_block_image]
    )

    assert await chat_message.atruncate(
        max_tokens=await tb.aestimate_tokens()
    ) == ChatMessage(
        blocks=[CitableBlock(title="Test Title", source="Test Source", content=[tb])]
    )
    assert await chat_message.atruncate(
        max_tokens=await tb.aestimate_tokens() + await ib.aestimate_tokens()
    ) == ChatMessage(
        blocks=[
            CitableBlock(title="Test Title", source="Test Source", content=[tb, ib])
        ]
    )
    assert await chat_message.atruncate(
        max_tokens=sum([await b.aestimate_tokens() for b in [tb, ib, db]])
    ) == ChatMessage(
        blocks=[
            CitableBlock(title="Test Title", source="Test Source", content=[tb, ib, db])
        ]
    )
    assert await chat_message.atruncate(
        max_tokens=sum([await b.aestimate_tokens() for b in [tb, ib, db, tb]])
    ) == ChatMessage(blocks=[citable_block, citation_block_text])
    assert (
        await chat_message.atruncate(max_tokens=await chat_message.aestimate_tokens())
        == chat_message
    )

    # reverse truncation
    assert await chat_message.atruncate(
        max_tokens=await ib.aestimate_tokens(), reverse=True
    ) == ChatMessage(blocks=[citation_block_image])
    assert await chat_message.atruncate(
        max_tokens=await ib.aestimate_tokens() + await tb.aestimate_tokens(),
        reverse=True,
    ) == ChatMessage(blocks=[citation_block_text, citation_block_image])
    assert await chat_message.atruncate(
        max_tokens=sum([await b.aestimate_tokens() for b in [ib, tb, db]]), reverse=True
    ) == ChatMessage(
        blocks=[
            CitableBlock(title="Test Title", source="Test Source", content=[db]),
            citation_block_text,
            citation_block_image,
        ]
    )
    assert await chat_message.atruncate(
        max_tokens=sum([await b.aestimate_tokens() for b in [ib, tb, db, ib]]),
        reverse=True,
    ) == ChatMessage(
        blocks=[
            CitableBlock(title="Test Title", source="Test Source", content=[ib, db]),
            citation_block_text,
            citation_block_image,
        ]
    )
    assert (
        await chat_message.atruncate(
            max_tokens=await chat_message.aestimate_tokens(), reverse=True
        )
        == chat_message
    )


@pytest.mark.asyncio
async def test_chat_message_amerge(
    png_1px,
    mp3_bytes,
    mp4_bytes,
    mock_pdf_bytes,
):
    m1 = ChatMessage(blocks=[TextBlock(text="Hello world!")])
    m2 = ChatMessage(blocks=[TextBlock(text="This is a test.")])
    m3 = ChatMessage(blocks=[ImageBlock(image=png_1px)])
    m4 = ChatMessage(blocks=[AudioBlock(audio=mp3_bytes)])
    m5 = ChatMessage(blocks=[AudioBlock(audio=mp3_bytes)])
    m6 = ChatMessage(blocks=[VideoBlock(video=mp4_bytes)])
    m7 = ChatMessage(blocks=[VideoBlock(video=mp4_bytes)])
    m8 = ChatMessage(blocks=[DocumentBlock(data=mock_pdf_bytes)])
    m9 = ChatMessage(blocks=[TextBlock(text="Hello human!")])
    m10 = ChatMessage(
        blocks=[TextBlock(text="This is another test.")], role=MessageRole.ASSISTANT
    )

    merged_m = await ChatMessage.amerge(
        [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10], chunk_size=10000
    )
    assert len(merged_m) == 2
    assert len(merged_m[0].blocks) == 8
    assert len(merged_m[1].blocks) == 1
    assert merged_m == [
        ChatMessage(
            # The first two text blocks are merged because they are consecutive
            blocks=await TextBlock.amerge(m1.blocks + m2.blocks, chunk_size=10000)
            # Image, Audio, Video, and Document blocks are not mergeable, but should be merged
            + m3.blocks
            + m4.blocks
            + m5.blocks
            + m6.blocks
            + m7.blocks
            + m8.blocks
            + m9.blocks
        ),
        # m10 has a different role, so it should be its own message, even though it's a consecutive TextBlock to m6
        ChatMessage(blocks=m10.blocks, role=MessageRole.ASSISTANT),
    ]


def test_chat_message_get_template_vars():
    chat_message = ChatMessage(
        blocks=[
            # non-recursive types
            TextBlock(text="{text}"),
            ImageBlock(image=b"{image_bytes}"),
            AudioBlock(audio=b"{audio_bytes}"),
            VideoBlock(video=b"{video_bytes}"),
            DocumentBlock(data=b"{pdf_bytes}"),
            # CachePoint, Thinking Blocks and Tool Call Blocks ot templatable, so we don't include them here
            # recursive types
            CitableBlock(
                title="Test Title",
                source="Test Source",
                content=[
                    TextBlock(text="{citable_text}"),
                    ImageBlock(image=b"{citable_image_bytes}"),
                    DocumentBlock(data=b"{citable_pdf_bytes}"),
                ],
            ),
            CitationBlock(
                title="Text Title",
                source="Text Source",
                cited_content=TextBlock(text="{citation_text}"),
                additional_location_info={},
            ),
            CitationBlock(
                title="Image Title",
                source="Image Source",
                cited_content=ImageBlock(image=b"{citation_image_bytes}"),
                additional_location_info={},
            ),
        ]
    )
    assert set(chat_message.get_template_vars()) == {
        "text",
        "image_bytes",
        "audio_bytes",
        "video_bytes",
        "pdf_bytes",
        "citable_text",
        "citable_image_bytes",
        "citable_pdf_bytes",
        "citation_text",
        "citation_image_bytes",
    }


def test_chat_message_format(
    png_1px,
    png_1px_b64,
    mp3_bytes,
    mp3_base64,
    mp4_bytes,
    mp4_base64,
    mock_pdf_bytes,
    pdf_base64,
):
    chat_message = ChatMessage(
        blocks=[
            # non-recursive types
            TextBlock(text="{text}"),
            ImageBlock(image=b"{image_bytes}"),
            AudioBlock(audio=b"{audio_bytes}"),
            VideoBlock(video=b"{video_bytes}"),
            DocumentBlock(data=b"{pdf_bytes}"),
            # CachePoint, Thinking Blocks and Tool Call Blocks ot templatable, so we don't include them here
            # recursive types
            CitableBlock(
                title="Test Title",
                source="Test Source",
                content=[
                    TextBlock(text="{citable_text}"),
                    ImageBlock(image=b"{citable_image_bytes}"),
                    DocumentBlock(data=b"{citable_pdf_bytes}"),
                ],
            ),
            CitationBlock(
                title="Text Title",
                source="Text Source",
                cited_content=TextBlock(text="{citation_text}"),
                additional_location_info={},
            ),
            CitationBlock(
                title="Image Title",
                source="Image Source",
                cited_content=ImageBlock(image=b"{citation_image_bytes}"),
                additional_location_info={},
            ),
        ]
    )

    formatted_message = chat_message.format_vars(
        text="Hello, world!",
        image_bytes=png_1px,
        audio_bytes=mp3_bytes,
        video_bytes=mp4_bytes,
        pdf_bytes=mock_pdf_bytes,
        citable_text="This is citable text.",
        citable_image_bytes=png_1px,
        citable_pdf_bytes=mock_pdf_bytes,
        citation_text="This is citation text.",
        citation_image_bytes=png_1px,
    )
    formatted_messageb64 = chat_message.format_vars(
        text="Hello, world!",
        image_bytes=png_1px_b64,
        audio_bytes=mp3_base64,
        video_bytes=mp4_base64,
        pdf_bytes=pdf_base64,
        citable_text="This is citable text.",
        citable_image_bytes=png_1px_b64,
        citable_pdf_bytes=pdf_base64,
        citation_text="This is citation text.",
        citation_image_bytes=png_1px_b64,
    )

    assert (
        formatted_message.blocks[0].text
        == formatted_messageb64.blocks[0].text
        == "Hello, world!"
    )
    assert (
        formatted_message.blocks[1].image
        == formatted_messageb64.blocks[1].image
        == png_1px_b64
    )
    assert (
        formatted_message.blocks[2].audio
        == formatted_messageb64.blocks[2].audio
        == mp3_base64
    )
    assert (
        formatted_message.blocks[3].video
        == formatted_messageb64.blocks[3].video
        == mp4_base64
    )
    assert (
        formatted_message.blocks[4].data
        == formatted_messageb64.blocks[4].data
        == pdf_base64
    )
    assert (
        formatted_message.blocks[5].nested_blocks[0].text
        == formatted_messageb64.blocks[5].nested_blocks[0].text
        == "This is citable text."
    )
    assert (
        formatted_message.blocks[5].nested_blocks[1].image
        == formatted_messageb64.blocks[5].nested_blocks[1].image
        == png_1px_b64
    )
    assert (
        formatted_message.blocks[5].nested_blocks[2].data
        == formatted_messageb64.blocks[5].nested_blocks[2].data
        == pdf_base64
    )
    assert (
        formatted_message.blocks[6].cited_content.text
        == formatted_messageb64.blocks[6].cited_content.text
        == "This is citation text."
    )
    assert (
        formatted_message.blocks[7].cited_content.image
        == formatted_messageb64.blocks[7].cited_content.image
        == png_1px_b64
    )


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


def test_text_block_get_template_vars():
    tb = TextBlock(text="Hello {addressee}!")
    vars = tb.get_template_vars()
    assert vars == ["addressee"]


def test_text_block_format():
    tb = TextBlock(text="Hello {addressee}!")
    formatted_tb = tb.format_vars(addressee="world")
    assert formatted_tb.text == "Hello world!"


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


def test_image_block_get_template_vars():
    ib = ImageBlock(image=b"{image_bytes}")
    assert ib.get_template_vars() == ["image_bytes"]


def test_image_block_format(png_1px: bytes, png_1px_b64: bytes):
    ib = ImageBlock(image=b"{image_bytes}")
    formatted_ib = ib.format_vars(image_bytes=png_1px)
    formatted_ibb64 = ib.format_vars(image_bytes=png_1px_b64)
    assert formatted_ib.image == png_1px_b64
    assert formatted_ibb64.image == png_1px_b64


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


def test_audio_block_get_template_vars():
    ab = AudioBlock(audio=b"{audio_bytes}")
    assert ab.get_template_vars() == ["audio_bytes"]


def test_audio_block_format(mp3_bytes: bytes, mp3_base64: bytes):
    ab = AudioBlock(audio=b"{audio_bytes}")
    formatted_ib_bytes = ab.format_vars(audio_bytes=mp3_bytes)
    assert formatted_ib_bytes.audio == mp3_base64


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


def test_video_block_get_template_vars():
    vb = VideoBlock(video=b"{video_bytes}")
    assert vb.get_template_vars() == ["video_bytes"]


def test_video_block_format(mp4_bytes: bytes, mp4_base64: bytes):
    vb = VideoBlock(video=b"{video_bytes}")
    formatted_vb = vb.format_vars(video_bytes=mp4_bytes)
    formatted_vbb64 = vb.format_vars(video_bytes=mp4_base64)
    assert formatted_vb.video == mp4_base64
    assert formatted_vbb64.video == mp4_base64


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


def test_document_block_get_template_vars():
    db = DocumentBlock(data=b"{pdf_bytes}", document_mimetype="application/pdf")
    assert db.get_template_vars() == ["pdf_bytes"]


def test_document_block_format(mock_pdf_bytes: bytes, pdf_base64: bytes):
    db = DocumentBlock(data=b"{pdf_bytes}", document_mimetype="application/pdf")
    formatted_db = db.format_vars(pdf_bytes=mock_pdf_bytes)
    formatted_dbb64 = db.format_vars(pdf_bytes=pdf_base64)
    assert formatted_db.data == pdf_base64
    assert formatted_dbb64.data == pdf_base64


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


def test_cache_control_get_template_vars():
    cp = CachePoint(cache_control=CacheControl(type="{cache_type}"))

    # CacheControl does not support template vars currently
    assert cp.get_template_vars() == []


def test_cache_control_format():
    cp = CachePoint(cache_control=CacheControl(type="{cache_type}"))
    formatted_cp = cp.format_vars(cache_type="ephemeral")

    # CacheControl does not support template vars currently
    assert formatted_cp.cache_control.type == "{cache_type}"


@pytest.mark.asyncio
async def test_citable_block_aestimate_tokens(png_1px: bytes, mock_pdf_bytes: bytes):
    content_blocks = [
        TextBlock(text="This is the content."),
        ImageBlock(image=png_1px),
        DocumentBlock(data=mock_pdf_bytes, document_mimetype="application/pdf"),
    ]
    cb = CitableBlock(title="Test Title", source="Test Source", content=content_blocks)
    assert await cb.aestimate_tokens() == sum(
        [await block.aestimate_tokens() for block in content_blocks]
    )


@pytest.mark.asyncio
async def test_citable_block_asplit(png_1px: bytes, mock_pdf_bytes: bytes):
    content_blocks = [
        TextBlock(text="This is the content."),
        ImageBlock(image=png_1px),
        DocumentBlock(data=mock_pdf_bytes, document_mimetype="application/pdf"),
    ]
    cb = CitableBlock(title="Test Title", source="Test Source", content=content_blocks)
    chunks = await cb.asplit(max_tokens=3)

    # Citable blocks are recursively splittable. However, since ImageBlock and DocumentBlock are not splittable, only
    # the TextBlock gets split. We expect 4 chunks: one for each original block.
    assert len(chunks) == 4
    assert chunks[0] == CitableBlock(
        title="Test Title",
        source="Test Source",
        content=[TextBlock(text="This is the")],
    )
    assert chunks[1] == CitableBlock(
        title="Test Title", source="Test Source", content=[TextBlock(text="content.")]
    )
    assert chunks[2] == CitableBlock(
        title="Test Title", source="Test Source", content=[ImageBlock(image=png_1px)]
    )
    assert chunks[3] == CitableBlock(
        title="Test Title",
        source="Test Source",
        content=[
            DocumentBlock(data=mock_pdf_bytes, document_mimetype="application/pdf")
        ],
    )


@pytest.mark.asyncio
async def test_citable_block_atruncate(png_1px: bytes, mock_pdf_bytes: bytes):
    tb = TextBlock(text="This is the content.")
    ib = ImageBlock(image=png_1px)
    db = DocumentBlock(data=mock_pdf_bytes, document_mimetype="application/pdf")
    cb = CitableBlock(title="Test Title", source="Test Source", content=[tb, ib, db])
    truncated_cb = await cb.atruncate(max_tokens=await tb.aestimate_tokens())
    truncated_cb_reverse = await cb.atruncate(
        max_tokens=await db.aestimate_tokens(), reverse=True
    )
    truncated_cb2 = await cb.atruncate(
        max_tokens=await tb.aestimate_tokens() + await ib.aestimate_tokens()
    )
    truncated_cb2_reverse = await cb.atruncate(
        max_tokens=await db.aestimate_tokens() + await ib.aestimate_tokens(),
        reverse=True,
    )
    truncated_cb3 = await cb.atruncate(
        max_tokens=sum([await b.aestimate_tokens() for b in [tb, ib, db]])
    )
    truncated_cb3_reverse = await cb.atruncate(
        max_tokens=sum([await b.aestimate_tokens() for b in [db, ib, tb]]), reverse=True
    )

    # Citable blocks are recursively truncatable. However, since ImageBlock and DocumentBlock are not truncatable,
    # only the TextBlock gets truncated.
    assert len(truncated_cb.content) == 1
    assert len(truncated_cb_reverse.content) == 1
    assert truncated_cb.content == [tb]
    assert truncated_cb_reverse.content == [db]

    # Truncation for recursive blocks will continue adding blocks until max_tokens is reached.
    assert len(truncated_cb2.content) == 2
    assert len(truncated_cb2_reverse.content) == 2
    assert truncated_cb2.content == [tb, ib]
    assert truncated_cb2_reverse.content == [ib, db]

    assert len(truncated_cb3.content) == 3
    assert len(truncated_cb3_reverse.content) == 3
    assert truncated_cb3.content == [tb, ib, db]
    assert truncated_cb3_reverse.content == [tb, ib, db]


@pytest.mark.asyncio
async def test_citable_block_amerge(png_1px: bytes, mock_pdf_bytes: bytes):
    content_blocks1 = [
        TextBlock(text="This is the content."),
        ImageBlock(image=png_1px),
    ]
    content_blocks2 = [
        DocumentBlock(data=mock_pdf_bytes, document_mimetype="application/pdf"),
        TextBlock(text="More content."),
    ]
    content_blocks3 = [
        TextBlock(text="This is also the content."),
        TextBlock(text="More content."),
    ]
    cb1 = CitableBlock(
        title="Test Title 1", source="Test Source 1", content=content_blocks1
    )
    cb2 = CitableBlock(
        title="Test Title 1", source="Test Source 1", content=content_blocks2
    )
    cb3 = CitableBlock(
        title="Test Title 2", source="Test Source 2", content=content_blocks3
    )
    merged_cbs = await CitableBlock.amerge([cb1, cb2, cb3], chunk_size=10000)

    # content of cb1 and cb2 should be merged, cb3 remains separate because it's of different title/source
    assert len(merged_cbs) == 2
    # first merged block should contain content from cb1 and cb2
    # The two TextBlocks are not merged since they are not consecutive in the original list
    assert merged_cbs[0].content == content_blocks1 + content_blocks2
    # Second merged block should be cb3 with its content merged since they are two consecutive TextBlocks
    assert merged_cbs[1].content == await TextBlock.amerge(
        content_blocks3, chunk_size=10000
    )


def test_citable_block_get_template_vars():
    content_blocks = [
        TextBlock(text="{text}"),
        ImageBlock(image=b"{image_bytes}"),
        DocumentBlock(data=b"{pdf_bytes}", document_mimetype="application/pdf"),
    ]
    cb = CitableBlock(title="Test Title", source="Test Source", content=content_blocks)
    assert set(cb.get_template_vars()) == {"text", "image_bytes", "pdf_bytes"}


def test_citable_block_format(
    png_1px: bytes, png_1px_b64: bytes, mock_pdf_bytes: bytes, pdf_base64: bytes
):
    content_blocks = [
        TextBlock(text="{text}"),
        ImageBlock(image=b"{image_bytes}"),
        DocumentBlock(data=b"{pdf_bytes}", document_mimetype="application/pdf"),
    ]
    cb = CitableBlock(title="Test Title", source="Test Source", content=content_blocks)
    formatted_cb = cb.format_vars(
        text="This is the content.", image_bytes=png_1px, pdf_bytes=mock_pdf_bytes
    )
    formatted_cbb64 = cb.format_vars(
        text="This is the content.", image_bytes=png_1px_b64, pdf_bytes=pdf_base64
    )
    assert formatted_cb.content[0] == TextBlock(text="This is the content.")
    assert formatted_cb.content[1] == ImageBlock(image=png_1px)
    assert formatted_cb.content[2] == DocumentBlock(
        data=mock_pdf_bytes, document_mimetype="application/pdf"
    )
    assert formatted_cbb64.content[0] == TextBlock(text="This is the content.")
    assert formatted_cbb64.content[1] == ImageBlock(image=png_1px)
    assert formatted_cbb64.content[2] == DocumentBlock(
        data=mock_pdf_bytes, document_mimetype="application/pdf"
    )


@pytest.mark.asyncio
async def test_citation_block_aestimate_tokens(png_1px):
    cb1 = CitationBlock(
        cited_content=TextBlock(text="Hello world! This is a test."),
        source="Test Source",
        title="Test Title",
        additional_location_info={},
    )
    cb2 = CitationBlock(
        cited_content=ImageBlock(image=png_1px),
        source="Test Source",
        title="Test Title",
        additional_location_info={},
    )
    assert await cb1.aestimate_tokens() == await cb1.cited_content.aestimate_tokens()
    assert await cb2.aestimate_tokens() == await cb2.cited_content.aestimate_tokens()


@pytest.mark.asyncio
async def test_citation_block_asplit(png_1px):
    cb1 = CitationBlock(
        cited_content=TextBlock(text="Hello world! This is a test."),
        source="Test Source",
        title="Test Title",
        additional_location_info={},
    )
    cb2 = CitationBlock(
        cited_content=ImageBlock(image=png_1px),
        source="Test Source",
        title="Test Title",
        additional_location_info={},
    )

    assert await cb1.asplit(max_tokens=3) == [
        CitationBlock(
            cited_content=chunk,
            source="Test Source",
            title="Test Title",
            additional_location_info={},
        )
        for chunk in await cb1.cited_content.asplit(max_tokens=3)
    ]
    assert await cb2.asplit(max_tokens=3) == [
        CitationBlock(
            cited_content=chunk,
            source="Test Source",
            title="Test Title",
            additional_location_info={},
        )
        for chunk in await cb2.cited_content.asplit(max_tokens=3)
    ]


@pytest.mark.asyncio
async def test_citation_block_atruncate(png_1px):
    cb1 = CitationBlock(
        cited_content=TextBlock(text="Hello world! This is a test."),
        source="Test Source",
        title="Test Title",
        additional_location_info={},
    )
    cb2 = CitationBlock(
        cited_content=ImageBlock(image=png_1px),
        source="Test Source",
        title="Test Title",
        additional_location_info={},
    )

    assert await cb1.atruncate(max_tokens=3) == CitationBlock(
        cited_content=await cb1.cited_content.atruncate(max_tokens=3),
        source="Test Source",
        title="Test Title",
        additional_location_info={},
    )
    assert await cb1.atruncate(max_tokens=3, reverse=True) == CitationBlock(
        cited_content=await cb1.cited_content.atruncate(max_tokens=3, reverse=True),
        source="Test Source",
        title="Test Title",
        additional_location_info={},
    )
    assert await cb2.atruncate(max_tokens=3) == CitationBlock(
        cited_content=await cb2.cited_content.atruncate(max_tokens=3),
        source="Test Source",
        title="Test Title",
        additional_location_info={},
    )
    assert await cb2.atruncate(max_tokens=3, reverse=True) == CitationBlock(
        cited_content=await cb2.cited_content.atruncate(max_tokens=3, reverse=True),
        source="Test Source",
        title="Test Title",
        additional_location_info={},
    )


@pytest.mark.asyncio
async def test_citation_block_amerge_text_blocks():
    cb1 = CitationBlock(
        cited_content=TextBlock(text="Hello world! "),
        source="Test Source",
        title="Test Title",
        additional_location_info={},
    )
    cb2 = CitationBlock(
        cited_content=TextBlock(text="This is a test."),
        source="Test Source",
        title="Test Title",
        additional_location_info={},
    )
    merged = await CitationBlock.amerge([cb1, cb2], chunk_size=100)

    # Both citation blocks should be merged into one
    assert len(merged) == 1
    assert merged[0] == CitationBlock(
        cited_content=(
            await TextBlock.amerge(
                [cb1.cited_content, cb2.cited_content], chunk_size=100
            )
        )[0],
        source="Test Source",
        title="Test Title",
        additional_location_info={},
    )


@pytest.mark.asyncio
async def test_citation_block_amerge_image_blocks(png_1px):
    cb1 = CitationBlock(
        cited_content=ImageBlock(image=png_1px),
        source="Test Source",
        title="Test Title",
        additional_location_info={},
    )
    cb2 = CitationBlock(
        cited_content=ImageBlock(image=png_1px),
        source="Test Source",
        title="Test Title",
        additional_location_info={},
    )

    # Image blocks are not mergeable currently
    assert await CitationBlock.amerge([cb1, cb2], chunk_size=100) == [cb1, cb2]


@pytest.mark.asyncio
async def test_citation_block_amerge_different_types(png_1px):
    cb1 = CitationBlock(
        cited_content=TextBlock(text="Hello world! This is a test."),
        source="Test Source",
        title="Test Title",
        additional_location_info={},
    )
    cb2 = CitationBlock(
        cited_content=ImageBlock(image=png_1px),
        source="Test Source",
        title="Test Title",
        additional_location_info={},
    )
    # Citation blocks are not mergeable across different cited content types
    assert await CitationBlock.amerge([cb1, cb2], chunk_size=100) == [cb1, cb2]


def test_citation_block_get_template_vars(png_1px):
    cb1 = CitationBlock(
        cited_content=TextBlock(text="{text}"),
        source="Test Source",
        title="Test Title",
        additional_location_info={},
    )
    cb2 = CitationBlock(
        cited_content=ImageBlock(image=b"{image_bytes}"),
        source="Test Source",
        title="Test Title",
        additional_location_info={},
    )

    assert cb1.get_template_vars() == ["text"]
    assert cb2.get_template_vars() == ["image_bytes"]


def test_citation_block_format(png_1px: bytes, png_1px_b64: bytes):
    cb1 = CitationBlock(
        cited_content=TextBlock(text="{text}"),
        source="Test Source",
        title="Test Title",
        additional_location_info={},
    )
    cb2 = CitationBlock(
        cited_content=ImageBlock(image=b"{image_bytes}"),
        source="Test Source",
        title="Test Title",
        additional_location_info={},
    )

    formatted_cb1 = cb1.format_vars(text="Hello world! This is a test.")
    formatted_cb2 = cb2.format_vars(image_bytes=png_1px)
    formatted_cb2b64 = cb2.format_vars(image_bytes=png_1px_b64)

    assert formatted_cb1.cited_content == TextBlock(text="Hello world! This is a test.")
    assert formatted_cb2.cited_content == ImageBlock(image=png_1px)
    assert formatted_cb2b64.cited_content == ImageBlock(image=png_1px)


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


def test_thinking_block_get_template_vars():
    block = ThinkingBlock(
        content="This is a {test} of the ThinkingBlock template vars."
    )
    # Currently, ThinkingBlock does not support template vars
    assert block.get_template_vars() == []


def test_thinking_block_format():
    block = ThinkingBlock(
        content="This is a {test} of the ThinkingBlock format method."
    )
    formatted_block = block.format_vars(test="demo")
    # Currently, ThinkingBlock does not support template vars, so content remains unchanged
    assert (
        formatted_block.content
        == "This is a {test} of the ThinkingBlock format method."
    )


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


def test_tool_call_block_get_template_vars():
    block = ToolCallBlock(
        tool_name="{tool_name}", tool_kwargs={"param": "{param_value}"}
    )
    # Currently, ToolCallBlock does not support template vars
    assert block.get_template_vars() == []


def test_tool_call_block_format():
    block = ToolCallBlock(
        tool_name="{tool_name}", tool_kwargs={"param": "{param_value}"}
    )
    formatted_block = block.format_vars(tool_name="example_tool", param_value="value1")

    # Currently, ToolCallBlock does not support template vars
    assert formatted_block.tool_name == "{tool_name}"
    assert formatted_block.tool_kwargs == {"param": "{param_value}"}
