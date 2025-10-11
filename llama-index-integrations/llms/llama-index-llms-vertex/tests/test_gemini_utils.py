import pytest
from google.cloud.aiplatform_v1beta1 import FunctionCall
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    TextBlock,
    ImageBlock,
)
from llama_index.llms.vertex.gemini_utils import (
    convert_chat_message_to_gemini_content,
    is_gemini_model,
)


def test_is_gemini_model():
    assert is_gemini_model("gemini-2.0-flash") is True
    assert is_gemini_model("chat-bison") is False


def test_convert_chat_message_to_gemini_content_with_function_call():
    message = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="",
        additional_kwargs={
            "tool_calls": [
                FunctionCall(
                    name="test_fn",
                    args={"arg1": "val1"},
                )
            ]
        },
    )

    result = convert_chat_message_to_gemini_content(message=message, is_history=True)

    assert result.role == "model"
    assert len(result.parts) == 1
    assert result.parts[0].function_call is not None
    assert result.parts[0].function_call.name == "test_fn"
    assert result.parts[0].function_call.args == {"arg1": "val1"}


def test_convert_chat_message_to_gemini_content_with_content():
    message = ChatMessage(
        role=MessageRole.USER,
        content="test content",
    )

    result = convert_chat_message_to_gemini_content(message=message, is_history=True)

    assert result.role == "user"
    assert result.text == "test content"
    assert len(result.parts) == 1
    assert result.parts[0].text == "test content"
    assert result.parts[0].function_call is None


def test_convert_chat_message_to_gemini_content_no_history():
    message = ChatMessage(
        role=MessageRole.USER,
        content="test content",
    )

    result = convert_chat_message_to_gemini_content(message=message, is_history=False)

    assert len(result) == 1
    assert result[0].text == "test content"
    assert result[0].function_call is None


def test_convert_chat_message_with_text_block():
    message = ChatMessage(
        role=MessageRole.USER,
        blocks=[TextBlock(text="Hello, world!")],
    )

    result = convert_chat_message_to_gemini_content(message=message, is_history=True)

    assert result.role == "user"
    assert len(result.parts) == 1
    assert result.parts[0].text == "Hello, world!"
    assert result.parts[0].function_call is None


def test_convert_chat_message_with_multiple_text_blocks():
    message = ChatMessage(
        role=MessageRole.USER,
        blocks=[
            TextBlock(text="Hi, "),
            TextBlock(text="there!"),
        ],
    )
    result = convert_chat_message_to_gemini_content(message=message, is_history=True)

    assert result.role == "user"
    assert len(result.parts) == 2
    assert result.parts[0].text == "Hi, "
    assert result.parts[1].text == "there!"


def test_convert_chat_message_with_empty_text_block():
    message = ChatMessage(
        role=MessageRole.USER,
        blocks=[TextBlock(text="")],
    )
    result = convert_chat_message_to_gemini_content(message=message, is_history=True)
    assert result.role == "user"
    assert len(result.parts) == 0


def test_convert_chat_message_with_invalid_image_block():
    message = ChatMessage(
        role=MessageRole.USER,
        blocks=[ImageBlock(path=None, image=None, url=None)],
    )
    with pytest.raises(
        ValueError, match="ImageBlock must have either path, url, or image data"
    ):
        convert_chat_message_to_gemini_content(message=message, is_history=True)
