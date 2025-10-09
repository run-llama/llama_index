import json
import base64
import pytest

# Use a relative import to refer to the local types.py file
from . import types

# --- Mock Object ---
class MockGoogleFunctionCall:
    """A mock object that mimics the structure of Google's FunctionCall."""

    def __init__(self, name: str, args: dict):
        self.name = name
        self.args = args

    def to_dict(self) -> dict:
        """The method our serializer fix relies on."""
        return {"name": self.name, "args": self.args}

    def __repr__(self) -> str:
        return f"<MockGoogleFunctionCall object name='{self.name}'>"


# --- Test Cases ---

def test_chat_message_with_google_function_call_serialization(monkeypatch):
    """
    Tests if a ChatMessage containing a mock Google FunctionCall object
    can be successfully serialized to JSON.
    """
    # 1. Arrange
    monkeypatch.setattr(types, "GOOGLE_FUNCTION_CALL_AVAILABLE", True)
    monkeypatch.setattr(types, "FunctionCall", MockGoogleFunctionCall)

    function_call_object = MockGoogleFunctionCall(
        name="get_current_weather",
        args={"location": "Boston, MA"},
    )
    message = types.ChatMessage(
        role=types.MessageRole.ASSISTANT,
        additional_kwargs={"tool_calls": [function_call_object]},
    )

    # 2. Act
    serialized_json = message.model_dump_json()

    # 3. Assert
    deserialized_data = json.loads(serialized_json)
    tool_calls = deserialized_data["additional_kwargs"]["tool_calls"]
    assert tool_calls[0]["name"] == "get_current_weather"

def test_chat_message_str_method():
    """Tests the string representation of a ChatMessage."""
    message = types.ChatMessage(role="user", content="Hello, world!")
    assert str(message) == "user: Hello, world!"

def test_chat_message_from_str():
    """Tests creating a ChatMessage using the from_str factory method."""
    # Test with default role
    message_user = types.ChatMessage.from_str("This is a test.")
    assert message_user.role == types.MessageRole.USER
    assert message_user.content == "This is a test."

    # Test with a specified string role
    message_asst = types.ChatMessage.from_str("I can help.", role="assistant")
    assert message_asst.role == types.MessageRole.ASSISTANT
    assert message_asst.content == "I can help."

def test_serialization_of_bytes_in_kwargs():
    """Tests if raw bytes in additional_kwargs are correctly base64 encoded."""
    raw_bytes = b"some binary data"
    message = types.ChatMessage(
        role="user",
        additional_kwargs={"data": raw_bytes}
    )
    
    # Act
    serialized_json = message.model_dump_json()
    deserialized_data = json.loads(serialized_json)

    # Assert
    expected_b64_string = base64.b64encode(raw_bytes).decode("utf-8")
    assert deserialized_data["additional_kwargs"]["data"] == expected_b64_string