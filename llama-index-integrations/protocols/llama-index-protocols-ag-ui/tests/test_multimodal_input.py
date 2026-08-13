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
)
from llama_index.core.base.llms.types import (
    AudioBlock,
    DocumentBlock,
    ImageBlock,
    TextBlock,
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


def test_audio_and_document_parts_convert() -> None:
    msg = ag_ui_message_to_llama_index_message(
        user_message(
            [
                AudioInputContent(
                    type="audio",
                    source=InputContentDataSource(
                        type="data", value=PNG_B64, mime_type="audio/mpeg"
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
    assert [type(b) for b in msg.blocks] == [AudioBlock, DocumentBlock]


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


def test_state_tag_is_stripped_from_text_part_of_multimodal_message() -> None:
    li_msg = ChatMessage(
        role="user",
        blocks=[
            TextBlock(text="<state>{'a': 1}</state>ask about the image"),
            ImageBlock(url="https://example.com/cat.png"),
        ],
        additional_kwargs={"id": "msg-2"},
    )
    back = llama_index_message_to_ag_ui_message(li_msg)
    assert isinstance(back.content, list)
    assert back.content[0].text == "ask about the image"
    assert back.content[1].type == "image"
