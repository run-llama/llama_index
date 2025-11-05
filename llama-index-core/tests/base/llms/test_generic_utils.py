"""Tests for llama_index.core.base.llms.generic_utils module."""

import pytest
from llama_index.core.base.llms.generic_utils import prompt_to_messages
from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock


def test_prompt_to_messages_with_string():
    """Test that prompt_to_messages correctly converts a string to messages."""
    prompt = "Hello, how are you?"
    messages = prompt_to_messages(prompt)

    assert len(messages) == 1
    assert isinstance(messages[0], ChatMessage)
    assert messages[0].role == MessageRole.USER
    assert messages[0].content == prompt
    assert len(messages[0].blocks) == 1
    assert isinstance(messages[0].blocks[0], TextBlock)
    assert messages[0].blocks[0].text == prompt


def test_prompt_to_messages_with_single_element_list_raises_error():
    """Test that prompt_to_messages raises TypeError for single-element list with helpful suggestion."""
    prompt = ["Hello, how are you?"]

    with pytest.raises(TypeError) as exc_info:
        prompt_to_messages(prompt)

    error_message = str(exc_info.value)
    assert "prompt must be a string" in error_message
    assert "got a list with 1 element" in error_message
    # Should provide helpful suggestion
    assert "Did you mean to pass" in error_message
    assert "'Hello, how are you?'" in error_message
    assert "chat() or stream_chat()" in error_message


def test_prompt_to_messages_with_multi_element_list_raises_error():
    """Test that prompt_to_messages raises TypeError for multi-element lists."""
    prompt = ["Hello", "How are you?"]

    with pytest.raises(TypeError) as exc_info:
        prompt_to_messages(prompt)

    error_message = str(exc_info.value)
    assert "prompt must be a string" in error_message
    assert "got a list with 2 elements" in error_message
    assert "chat() or stream_chat()" in error_message


def test_prompt_to_messages_with_list_of_non_strings_raises_error():
    """Test that prompt_to_messages raises TypeError for lists with non-string elements."""
    prompt = [{"content": "Hello"}]

    with pytest.raises(TypeError) as exc_info:
        prompt_to_messages(prompt)

    error_message = str(exc_info.value)
    assert "prompt must be a string" in error_message
    assert "chat() or stream_chat()" in error_message


def test_prompt_to_messages_with_dict_raises_error():
    """Test that prompt_to_messages raises TypeError for dict input."""
    prompt = {"role": "user", "content": "Hello"}

    with pytest.raises(TypeError) as exc_info:
        prompt_to_messages(prompt)

    error_message = str(exc_info.value)
    assert "prompt must be a string" in error_message
    assert "got dict" in error_message
    assert "chat() or stream_chat()" in error_message


def test_prompt_to_messages_with_chat_message_raises_error():
    """Test that prompt_to_messages raises TypeError for ChatMessage input."""
    prompt = ChatMessage(role=MessageRole.USER, content="Hello")

    with pytest.raises(TypeError) as exc_info:
        prompt_to_messages(prompt)

    error_message = str(exc_info.value)
    assert "prompt must be a string" in error_message
    assert "got ChatMessage" in error_message
    assert "chat() or stream_chat()" in error_message


def test_prompt_to_messages_with_none_raises_error():
    """Test that prompt_to_messages raises TypeError for None input."""
    with pytest.raises(TypeError) as exc_info:
        prompt_to_messages(None)

    error_message = str(exc_info.value)
    assert "prompt must be a string" in error_message
    assert "got NoneType" in error_message


def test_prompt_to_messages_with_empty_string():
    """Test that prompt_to_messages handles empty strings correctly."""
    prompt = ""
    messages = prompt_to_messages(prompt)

    assert len(messages) == 1
    assert isinstance(messages[0], ChatMessage)
    assert messages[0].role == MessageRole.USER
    assert messages[0].content == ""
    assert len(messages[0].blocks) == 1
    assert isinstance(messages[0].blocks[0], TextBlock)
    assert messages[0].blocks[0].text == ""


def test_prompt_to_messages_error_message_is_helpful():
    """Test that error messages provide clear guidance on correct usage."""
    # This reproduces the exact issue from bug report #20215
    # Single-element list should give helpful error with suggestion
    prompt = ["What is the capital of France? answer short."]

    with pytest.raises(TypeError) as exc_info:
        prompt_to_messages(prompt)

    error_message = str(exc_info.value)
    # Should suggest the correct way to pass the string
    assert "Did you mean to pass" in error_message
    assert "'What is the capital of France? answer short.'" in error_message

    # Multi-element list should give helpful error
    prompt_multi = ["Question 1", "Question 2"]
    with pytest.raises(TypeError) as exc_info:
        prompt_to_messages(prompt_multi)

    error_message = str(exc_info.value)
    # Verify the error message is helpful and directs to the right methods
    assert "got a list with 2 element" in error_message
    assert "chat() or stream_chat()" in error_message
