"""The state prompt must not crash on (or drop media from) multimodal input.

Regression test for the ``msg.content = ...`` assignment in
``AGUIChatWorkflow``: the ``ChatMessage.content`` setter raises on a
multi-block message, so a user message carrying an image alongside text made
every run with non-empty state fail before the blocks-preserving rewrite.
"""

from llama_index.core.base.llms.types import ImageBlock, TextBlock
from llama_index.core.llms import ChatMessage

from llama_index.protocols.ag_ui.agent import DEFAULT_STATE_PROMPT


def rewrite_last_user_message(chat_history, state) -> None:
    """Mirror of the state-prompt rewrite in AGUIChatWorkflow.prepare_chat."""
    for msg in chat_history[::-1]:
        if msg.role.value == "user":
            state_text = DEFAULT_STATE_PROMPT.format(
                state=str(state), user_input=msg.content or ""
            )
            msg.blocks = [
                TextBlock(text=state_text),
                *[block for block in msg.blocks if not isinstance(block, TextBlock)],
            ]
            break


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
    rewrite_last_user_message(history, {"a": 1})
    blocks = history[0].blocks
    assert [type(b) for b in blocks] == [TextBlock, ImageBlock]
    assert "what is in this image?" in blocks[0].text
    assert "{'a': 1}" in blocks[0].text
    assert str(blocks[1].url) == "https://example.com/cat.png"


def test_state_prompt_still_works_for_text_only() -> None:
    history = [ChatMessage(role="user", content="hello")]
    rewrite_last_user_message(history, {"a": 1})
    assert "hello" in history[0].content
    assert "{'a': 1}" in history[0].content
