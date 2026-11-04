"""Tests for Gemini message utilities."""

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.utilities.gemini_utils import (
    merge_neighboring_same_role_messages,
)


def test_merge_neighboring_same_role_messages_does_not_mutate_input() -> None:
    """
    Merging must not mutate the caller's input messages.

    Regression: merged_content aliased current_message.blocks, so extend() mutated
    the caller's first message in place. Re-running the merge on the same list then
    produced duplicated blocks.
    """
    messages = [
        ChatMessage(role=MessageRole.USER, content="U1"),
        ChatMessage(role=MessageRole.USER, content="U2"),
        ChatMessage(role=MessageRole.ASSISTANT, content="A1"),
    ]

    merged = merge_neighboring_same_role_messages(messages)

    # The two neighboring USER messages merge into one.
    assert [m.role for m in merged] == [MessageRole.USER, MessageRole.MODEL]
    assert [b.text for b in merged[0].blocks] == ["U1", "U2"]

    # The caller's input is untouched.
    assert [b.text for b in messages[0].blocks] == ["U1"]
    assert [b.text for b in messages[1].blocks] == ["U2"]

    # Merging the same input again is idempotent (not corrupted by the first call).
    merged_again = merge_neighboring_same_role_messages(messages)
    assert [b.text for b in merged_again[0].blocks] == ["U1", "U2"]
