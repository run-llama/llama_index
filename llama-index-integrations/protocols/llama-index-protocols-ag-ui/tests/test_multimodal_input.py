import base64

from ag_ui.core import UserMessage
from ag_ui.core.types import (
    AudioInputContent,
    BinaryInputContent,
    DocumentInputContent,
    ImageInputContent,
    InputContentDataSource,
    InputContentUrlSource,
    TextInputContent,
    VideoInputContent,
)
from llama_index.core.base.llms.types import (
    AudioBlock,
    DocumentBlock,
    ImageBlock,
    TextBlock,
    VideoBlock,
)
from llama_index.core.llms import ChatMessage

from llama_index.protocols.ag_ui.utils import (
    ag_ui_message_to_llama_index_message,
    agui_content_to_blocks,
    llama_index_message_to_ag_ui_message,
)

PNG_BYTES = b"\x89PNG\r\n\x1a\n fake image payload"
PNG_B64 = base64.b64encode(PNG_BYTES).decode("ascii")


def user_message(content) -> UserMessage:
    return UserMessage(id="msg-1", role="user", content=content)


def test_string_content_still_converts_to_text() -> None:
    msg = ag_ui_message_to_llama_index_message(user_message("hello"))
    assert msg.role.value == "user"
    assert msg.content == "hello"
    assert [type(b) for b in msg.blocks] == [TextBlock]


def test_image_data_part_becomes_image_block_with_bytes() -> None:
    msg = ag_ui_message_to_llama_index_message(
        user_message(
            [
                TextInputContent(type="text", text="what is in this image?"),
                ImageInputContent(
                    type="image",
                    source=InputContentDataSource(
                        type="data", value=PNG_B64, mime_type="image/png"
                    ),
                ),
            ]
        )
    )
    assert [type(b) for b in msg.blocks] == [TextBlock, ImageBlock]
    image = msg.blocks[1]
    assert image.resolve_image().read() == PNG_BYTES
    assert image.image_mimetype == "image/png"
    # The text is still reachable through the plain-content view.
    assert msg.content == "what is in this image?"


def test_image_url_part_becomes_image_block_with_url() -> None:
    msg = ag_ui_message_to_llama_index_message(
        user_message(
            [
                ImageInputContent(
                    type="image",
                    source=InputContentUrlSource(
                        type="url", value="https://example.com/cat.png"
                    ),
                )
            ]
        )
    )
    assert [type(b) for b in msg.blocks] == [ImageBlock]
    assert str(msg.blocks[0].url) == "https://example.com/cat.png"


def test_audio_video_and_document_parts_convert() -> None:
    msg = ag_ui_message_to_llama_index_message(
        user_message(
            [
                AudioInputContent(
                    type="audio",
                    source=InputContentDataSource(
                        type="data", value=PNG_B64, mime_type="audio/mpeg"
                    ),
                ),
                VideoInputContent(
                    type="video",
                    source=InputContentUrlSource(
                        type="url",
                        value="https://example.com/clip.mp4",
                        mime_type="video/mp4",
                    ),
                ),
                DocumentInputContent(
                    type="document",
                    source=InputContentUrlSource(
                        type="url",
                        value="https://example.com/report.pdf",
                        mime_type="application/pdf",
                    ),
                ),
            ]
        )
    )
    assert [type(b) for b in msg.blocks] == [AudioBlock, VideoBlock, DocumentBlock]
    # MIME carries through: audio/mpeg maps onto the block's format field, and
    # the audio bytes decode exactly once.
    assert msg.blocks[0].format == "mp3"
    assert msg.blocks[0].resolve_audio().read() == PNG_BYTES
    assert msg.blocks[1].video_mimetype == "video/mp4"


def test_audio_format_roundtrips_to_mime() -> None:
    original = user_message(
        [
            AudioInputContent(
                type="audio",
                source=InputContentDataSource(
                    type="data", value=PNG_B64, mime_type="audio/mpeg"
                ),
            )
        ]
    )
    back = llama_index_message_to_ag_ui_message(
        ag_ui_message_to_llama_index_message(original)
    )
    assert back.content[0].source.mime_type == "audio/mpeg"


def test_base64_lookalike_payload_is_not_double_decoded() -> None:
    # b"test" is itself a valid base64 string, the worst case for llama-index's
    # keep-if-already-base64 normalizer: decoding the AG-UI payload up front
    # would make resolve_*() decode it a second time and corrupt it.
    payload = b"test"
    encoded = base64.b64encode(payload).decode("ascii")
    msg = ag_ui_message_to_llama_index_message(
        user_message(
            [
                DocumentInputContent(
                    type="document",
                    source=InputContentDataSource(
                        type="data", value=encoded, mime_type="text/plain"
                    ),
                ),
                ImageInputContent(
                    type="image",
                    source=InputContentDataSource(
                        type="data", value=encoded, mime_type="image/png"
                    ),
                ),
            ]
        )
    )
    assert msg.blocks[0].resolve_document().read() == payload
    assert msg.blocks[1].resolve_image().read() == payload
    # And it survives the full roundtrip back to AG-UI parts.
    back = llama_index_message_to_ag_ui_message(msg)
    assert base64.b64decode(back.content[0].source.value) == payload
    assert base64.b64decode(back.content[1].source.value) == payload


def test_deprecated_binary_part_routes_by_mime() -> None:
    blocks = agui_content_to_blocks(
        [
            BinaryInputContent(
                type="binary", mime_type="image/jpeg", data=PNG_B64, filename="a.jpg"
            )
        ]
    )
    assert [type(b) for b in blocks] == [ImageBlock]


def test_unreadable_parts_are_skipped_not_stringified() -> None:
    blocks = agui_content_to_blocks(
        [
            TextInputContent(type="text", text="hi"),
            # id-only binary part: nothing to read, so it must be skipped.
            BinaryInputContent(type="binary", mime_type="image/png", id="file-123"),
        ]
    )
    assert [type(b) for b in blocks] == [TextBlock]


def test_roundtrip_multimodal_user_message() -> None:
    original = user_message(
        [
            TextInputContent(type="text", text="describe this"),
            ImageInputContent(
                type="image",
                source=InputContentDataSource(
                    type="data", value=PNG_B64, mime_type="image/png"
                ),
            ),
        ]
    )
    li_msg = ag_ui_message_to_llama_index_message(original)
    back = llama_index_message_to_ag_ui_message(li_msg)
    assert isinstance(back, UserMessage)
    assert isinstance(back.content, list)
    assert [p.type for p in back.content] == ["text", "image"]
    assert back.content[0].text == "describe this"
    assert base64.b64decode(back.content[1].source.value) == PNG_BYTES


def test_text_only_user_message_roundtrips_as_string() -> None:
    li_msg = ag_ui_message_to_llama_index_message(user_message("plain text"))
    back = llama_index_message_to_ag_ui_message(li_msg)
    assert back.content == "plain text"


def test_text_only_parts_array_roundtrips_as_parts() -> None:
    original = user_message(
        [
            TextInputContent(type="text", text="alpha"),
            TextInputContent(type="text", text="beta"),
        ]
    )
    back = llama_index_message_to_ag_ui_message(
        ag_ui_message_to_llama_index_message(original)
    )
    assert isinstance(back.content, list)
    assert [p.text for p in back.content] == ["alpha", "beta"]


def test_reverse_conversion_preserves_significant_whitespace() -> None:
    li_msg = ChatMessage(
        role="user",
        blocks=[
            TextBlock(text="    indented()\n"),
            ImageBlock(url="https://example.com/cat.png"),
        ],
        additional_kwargs={"id": "msg-3"},
    )
    back = llama_index_message_to_ag_ui_message(li_msg)
    assert back.content[0].text == "    indented()\n"


def test_parameterized_audio_mime_maps_to_bare_format() -> None:
    msg = ag_ui_message_to_llama_index_message(
        user_message(
            [
                AudioInputContent(
                    type="audio",
                    source=InputContentDataSource(
                        type="data", value=PNG_B64, mime_type="audio/wav; codecs=1"
                    ),
                )
            ]
        )
    )
    assert msg.blocks[0].format == "wav"


def test_user_authored_state_markup_survives_parts_roundtrip() -> None:
    original = user_message(
        [
            TextInputContent(
                type="text", text="explain before <state>ready</state> after"
            ),
            ImageInputContent(
                type="image",
                source=InputContentUrlSource(
                    type="url", value="https://example.com/xml.png"
                ),
            ),
        ]
    )
    back = llama_index_message_to_ag_ui_message(
        ag_ui_message_to_llama_index_message(original)
    )
    # No state was injected, so nothing may be stripped — even text that
    # happens to contain <state> markup.
    assert back.content[0].text == "explain before <state>ready</state> after"


def test_single_element_text_array_keeps_shape_and_whitespace() -> None:
    original = user_message([TextInputContent(type="text", text="    code()\n")])
    back = llama_index_message_to_ag_ui_message(
        ag_ui_message_to_llama_index_message(original)
    )
    assert isinstance(back.content, list)
    assert back.content[0].text == "    code()\n"


def test_empty_parts_array_roundtrips_empty() -> None:
    back = llama_index_message_to_ag_ui_message(
        ag_ui_message_to_llama_index_message(user_message([]))
    )
    assert back.content == []


def test_mime_types_are_case_insensitive() -> None:
    blocks = agui_content_to_blocks(
        [
            AudioInputContent(
                type="audio",
                source=InputContentDataSource(
                    type="data", value=PNG_B64, mime_type="Audio/MPEG"
                ),
            ),
            BinaryInputContent(type="binary", mime_type="Image/PNG", data=PNG_B64),
        ]
    )
    assert [type(b) for b in blocks] == [AudioBlock, ImageBlock]
    assert blocks[0].format == "mp3"
