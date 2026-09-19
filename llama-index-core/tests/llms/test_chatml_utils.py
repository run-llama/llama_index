from llama_index.core.base.llms.types import (
    ChatMessage,
    ImageBlock,
    MessageRole,
    ToolCallBlock,
)
from llama_index.core.llms.chatml_utils import messages_to_prompt


def test_messages_to_prompt_renders_plain_conversation():
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are helpful."),
        ChatMessage(role=MessageRole.USER, content="Hello"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Hi there"),
        ChatMessage(role=MessageRole.USER, content="How are you?"),
    ]

    prompt = messages_to_prompt(messages)

    assert (
        prompt == "<|im_start|>system\nYou are helpful. <|im_end|>\n"
        "<|im_start|>user\nHello <|im_end|>\n"
        "<|im_start|>assistant\nHi there <|im_end|>\n"
        "<|im_start|>user\nHow are you? <|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_messages_to_prompt_does_not_inject_literal_none_for_tool_calls():
    """An assistant tool-call turn has content=None; it must not be rendered as "None"."""
    messages = [
        ChatMessage(role=MessageRole.USER, content="Weather in Paris?"),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            blocks=[
                ToolCallBlock(
                    tool_call_id="c1",
                    tool_name="get_weather",
                    tool_kwargs={"city": "Paris"},
                )
            ],
        ),
        ChatMessage(role=MessageRole.USER, content="Thanks!"),
    ]

    prompt = messages_to_prompt(messages)

    assert "None" not in prompt
    assert "<|im_start|>assistant\n <|im_end|>\n" in prompt


def test_messages_to_prompt_does_not_inject_literal_none_for_block_only_user_message():
    """A user message carrying only non-text blocks has content=None as well."""
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            blocks=[ImageBlock(image=b"png-bytes", image_mimetype="image/png")],
        ),
    ]

    prompt = messages_to_prompt(messages)

    assert "None" not in prompt
