"""
The state prompt must not crash on (or corrupt) multimodal user messages.

Regression tests for the state-prompt rewrite in ``AGUIChatWorkflow``: the
``ChatMessage.content`` setter raises on a multi-block message, so a user
message carrying an image alongside text made every run with non-empty state
fail. Exercises the production ``_apply_state_prompt`` directly.
"""

from llama_index.core.base.llms.types import ImageBlock, TextBlock
from llama_index.core.llms import ChatMessage

from llama_index.protocols.ag_ui.agent import _apply_state_prompt


def test_state_prompt_preserves_image_blocks() -> None:
    history = [
        ChatMessage(
            role="user",
            blocks=[
                TextBlock(text="what is in this image?"),
                ImageBlock(url="https://example.com/cat.png"),
            ],
        )
    ]
    _apply_state_prompt(history, {"a": 1})
    blocks = history[0].blocks
    assert [type(b) for b in blocks] == [TextBlock, TextBlock, ImageBlock]
    assert "{'a': 1}" in blocks[0].text
    assert blocks[1].text == "what is in this image?"
    assert str(blocks[2].url) == "https://example.com/cat.png"


def test_state_prompt_preserves_interleaved_ordering() -> None:
    history = [
        ChatMessage(
            role="user",
            blocks=[
                TextBlock(text="first, this photo:"),
                ImageBlock(url="https://example.com/one.png"),
                TextBlock(text="and compare it with this one:"),
                ImageBlock(url="https://example.com/two.png"),
            ],
        )
    ]
    _apply_state_prompt(history, {"a": 1})
    blocks = history[0].blocks
    # State block is prepended; the original interleaving stays intact so each
    # caption keeps pointing at its own image.
    assert [type(b) for b in blocks] == [
        TextBlock,
        TextBlock,
        ImageBlock,
        TextBlock,
        ImageBlock,
    ]
    assert "{'a': 1}" in blocks[0].text
    assert blocks[1].text == "first, this photo:"
    assert str(blocks[2].url) == "https://example.com/one.png"
    assert blocks[3].text == "and compare it with this one:"
    assert str(blocks[4].url) == "https://example.com/two.png"


def test_state_prompt_text_only_unchanged_behavior() -> None:
    history = [ChatMessage(role="user", content="hello")]
    _apply_state_prompt(history, {"a": 1})
    assert len(history[0].blocks) == 1
    assert "hello" in history[0].content
    assert "{'a': 1}" in history[0].content


def test_state_prompt_keeps_text_parts_array_boundaries() -> None:
    history = [
        ChatMessage(
            role="user",
            blocks=[TextBlock(text="alpha"), TextBlock(text="beta")],
        )
    ]
    _apply_state_prompt(history, {"a": 1})
    blocks = history[0].blocks
    # A parts-array client's fragment boundaries survive: state is its own
    # leading block, the original text blocks stay separate and verbatim.
    assert [type(b) for b in blocks] == [TextBlock, TextBlock, TextBlock]
    assert "{'a': 1}" in blocks[0].text
    assert blocks[1].text == "alpha"
    assert blocks[2].text == "beta"


def test_state_prompt_respects_single_element_parts_marker() -> None:
    from llama_index.protocols.ag_ui.utils import (
        AG_UI_PARTS_KEY,
        ag_ui_message_to_llama_index_message,
    )
    from ag_ui.core import UserMessage
    from ag_ui.core.types import TextInputContent

    history = [
        ag_ui_message_to_llama_index_message(
            UserMessage(
                id="m1",
                role="user",
                content=[TextInputContent(type="text", text="    code()\n")],
            )
        )
    ]
    assert history[0].additional_kwargs.get(AG_UI_PARTS_KEY) is True
    _apply_state_prompt(history, {"a": 1})
    blocks = history[0].blocks
    # The single part is not folded into the template; it stays verbatim
    # behind the injected state block.
    assert len(blocks) == 2
    assert "{'a': 1}" in blocks[0].text
    assert blocks[1].text == "    code()\n"


def test_state_prompt_targets_latest_user_message() -> None:
    history = [
        ChatMessage(role="user", content="older question"),
        ChatMessage(role="assistant", content="answer"),
        ChatMessage(role="user", content="newest question"),
    ]
    _apply_state_prompt(history, {"a": 1})
    assert "{'a': 1}" not in history[0].content
    assert "{'a': 1}" in history[2].content


def test_injected_state_block_is_dropped_on_reverse_conversion() -> None:
    from ag_ui.core import UserMessage
    from ag_ui.core.types import (
        ImageInputContent,
        InputContentUrlSource,
        TextInputContent,
    )
    from llama_index.protocols.ag_ui.utils import (
        ag_ui_message_to_llama_index_message,
        llama_index_message_to_ag_ui_message,
    )

    history = [
        ag_ui_message_to_llama_index_message(
            UserMessage(
                id="m1",
                role="user",
                content=[
                    TextInputContent(type="text", text="ask about the image"),
                    ImageInputContent(
                        type="image",
                        source=InputContentUrlSource(
                            type="url", value="https://example.com/cat.png"
                        ),
                    ),
                ],
            )
        )
    ]
    _apply_state_prompt(history, {"a": 1})
    back = llama_index_message_to_ag_ui_message(history[0])
    # Exactly the injected state block is gone; the client's parts survive.
    assert isinstance(back.content, list)
    assert [p.type for p in back.content] == ["text", "image"]
    assert back.content[0].text == "ask about the image"
