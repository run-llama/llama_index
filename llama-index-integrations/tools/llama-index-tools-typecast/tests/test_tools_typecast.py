import pytest
from unittest.mock import Mock, patch, mock_open
from llama_index.tools.typecast import TypecastToolSpec


def test_class_inheritance():
    """Test that TypecastToolSpec inherits from BaseToolSpec"""
    from llama_index.core.tools.tool_spec.base import BaseToolSpec

    names_of_base_classes = [b.__name__ for b in TypecastToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_spec_functions():
    """Test that required functions are defined"""
    assert "get_voices" in TypecastToolSpec.spec_functions
    assert "text_to_speech" in TypecastToolSpec.spec_functions


def test_initialization():
    """Test tool initialization"""
    tool = TypecastToolSpec(api_key="test-key", host="https://test.api")
    assert tool.api_key == "test-key"
    assert tool.host == "https://test.api"


@patch("typecast.client.Typecast")
def test_get_voices_success(mock_typecast):
    """Test successful voice retrieval"""
    # Mock response
    mock_voice = Mock()
    mock_voice.model_dump.return_value = {"id": "voice1", "name": "Test Voice"}
    mock_typecast.return_value.voices.return_value = [mock_voice]

    tool = TypecastToolSpec(api_key="test-key")
    voices = tool.get_voices()

    assert len(voices) == 1
    assert voices[0]["name"] == "Test Voice"


@patch("typecast.client.Typecast")
def test_get_voices_failure(mock_typecast):
    """Test voice retrieval failure handling"""
    mock_typecast.return_value.voices.side_effect = Exception("API Error")

    tool = TypecastToolSpec(api_key="test-key")

    with pytest.raises(Exception) as exc_info:
        tool.get_voices()

    assert "Failed to get voices" in str(exc_info.value)


@patch("typecast.client.Typecast")
@patch("builtins.open", new_callable=mock_open)
def test_text_to_speech_success(mock_file, mock_typecast):
    """Test successful text-to-speech conversion"""
    # Mock response
    mock_response = Mock()
    mock_response.audio_data = b"audio data"
    mock_typecast.return_value.text_to_speech.return_value = mock_response

    tool = TypecastToolSpec(api_key="test-key")
    output_path = tool.text_to_speech(
        text="Hello world", voice_id="voice1", output_path="output.wav"
    )

    assert output_path == "output.wav"
    mock_file.assert_called_once_with("output.wav", "wb")


def test_text_to_speech_validation():
    """Test parameter validation"""
    tool = TypecastToolSpec(api_key="test-key")

    # Empty text
    with pytest.raises(ValueError, match="Text cannot be empty"):
        tool.text_to_speech(text="", voice_id="voice1", output_path="out.wav")

    # Missing voice_id
    with pytest.raises(ValueError, match="Voice ID is required"):
        tool.text_to_speech(text="Hello", voice_id="", output_path="out.wav")
