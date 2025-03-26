from google.cloud.aiplatform_v1beta1 import FunctionCall
from llama_index.core.base.llms.types import ChatMessage, MessageRole

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
