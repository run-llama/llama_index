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
    assert "get_voice" in TypecastToolSpec.spec_functions
    assert "text_to_speech" in TypecastToolSpec.spec_functions


def test_initialization():
    """Test tool initialization"""
    tool = TypecastToolSpec(api_key="test-key", host="https://test.api")
    assert tool.api_key == "test-key"
    assert tool.host == "https://test.api"


@patch("llama_index.tools.typecast.base.Typecast")
def test_get_voices_success(mock_typecast):
    """Test successful voice retrieval (V2 API)"""
    # Mock response with V2 structure
    mock_voice = Mock()
    mock_voice.model_dump.return_value = {
        "voice_id": "tc_123",
        "voice_name": "Test Voice",
        "models": [{"version": "ssfm-v21", "emotions": ["normal", "happy"]}],
        "gender": "female",
        "age": "young_adult",
        "use_cases": ["Audiobook", "Podcast"],
    }
    mock_typecast.return_value.voices_v2.return_value = [mock_voice]

    tool = TypecastToolSpec(api_key="test-key")
    voices = tool.get_voices()

    assert len(voices) == 1
    assert voices[0]["voice_name"] == "Test Voice"
    assert voices[0]["models"][0]["version"] == "ssfm-v21"
    mock_typecast.return_value.voices_v2.assert_called_once_with(filter=None)


@patch("llama_index.tools.typecast.base.Typecast")
@patch("llama_index.tools.typecast.base.VoicesV2Filter")
def test_get_voices_with_filters(mock_filter, mock_typecast):
    """Test voice retrieval with V2 filters"""
    mock_voice = Mock()
    mock_voice.model_dump.return_value = {
        "voice_id": "tc_123",
        "voice_name": "Test Voice",
        "models": [{"version": "ssfm-v30", "emotions": ["normal", "happy"]}],
        "gender": "female",
        "age": "young_adult",
    }
    mock_typecast.return_value.voices_v2.return_value = [mock_voice]

    tool = TypecastToolSpec(api_key="test-key")
    voices = tool.get_voices(model="ssfm-v30", gender="female", age="young_adult")

    assert len(voices) == 1
    mock_typecast.return_value.voices_v2.assert_called_once()


@patch("llama_index.tools.typecast.base.Typecast")
def test_get_voices_failure(mock_typecast):
    """Test voice retrieval failure handling"""
    from typecast.exceptions import TypecastError

    mock_typecast.return_value.voices_v2.side_effect = TypecastError("API Error")

    tool = TypecastToolSpec(api_key="test-key")

    with pytest.raises(TypecastError) as exc_info:
        tool.get_voices()

    assert "API Error" in str(exc_info.value)


@patch("llama_index.tools.typecast.base.Typecast")
def test_get_voice_success(mock_typecast):
    """Test successful single voice retrieval (V2 API)"""
    # Mock response with V2 structure
    mock_voice = Mock()
    mock_voice.model_dump.return_value = {
        "voice_id": "tc_123",
        "voice_name": "Test Voice",
        "models": [
            {"version": "ssfm-v21", "emotions": ["normal", "happy", "sad"]},
            {"version": "ssfm-v30", "emotions": ["normal", "happy", "sad", "whisper"]},
        ],
        "gender": "female",
        "age": "young_adult",
        "use_cases": ["Audiobook", "Podcast"],
    }
    mock_typecast.return_value.voice_v2.return_value = mock_voice

    tool = TypecastToolSpec(api_key="test-key")
    voice = tool.get_voice("tc_123")

    assert voice["voice_id"] == "tc_123"
    assert voice["voice_name"] == "Test Voice"
    assert len(voice["models"]) == 2
    assert "happy" in voice["models"][0]["emotions"]


@patch("llama_index.tools.typecast.base.Typecast")
def test_get_voice_not_found(mock_typecast):
    """Test voice not found handling"""
    from typecast.exceptions import NotFoundError

    mock_typecast.return_value.voice_v2.side_effect = NotFoundError("Voice not found")

    tool = TypecastToolSpec(api_key="test-key")

    with pytest.raises(NotFoundError) as exc_info:
        tool.get_voice("invalid_id")

    assert "Voice not found" in str(exc_info.value)


@patch("llama_index.tools.typecast.base.Typecast")
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


@patch("llama_index.tools.typecast.base.Typecast")
@patch("builtins.open", new_callable=mock_open)
def test_text_to_speech_with_seed(mock_file, mock_typecast):
    """Test text-to-speech conversion with seed parameter"""
    # Mock response
    mock_response = Mock()
    mock_response.audio_data = b"audio data"
    mock_typecast.return_value.text_to_speech.return_value = mock_response

    tool = TypecastToolSpec(api_key="test-key")
    output_path = tool.text_to_speech(
        text="Hello world",
        voice_id="voice1",
        output_path="output.wav",
        seed=42,
    )

    assert output_path == "output.wav"
    # Verify that the request was called with seed parameter
    call_args = mock_typecast.return_value.text_to_speech.call_args
    assert call_args is not None
    request = call_args[0][0]
    assert request.seed == 42


@patch("llama_index.tools.typecast.base.Typecast")
@patch("builtins.open", new_callable=mock_open)
def test_text_to_speech_with_all_parameters(mock_file, mock_typecast):
    """Test text-to-speech conversion with all parameters"""
    # Mock response
    mock_response = Mock()
    mock_response.audio_data = b"audio data"
    mock_typecast.return_value.text_to_speech.return_value = mock_response

    tool = TypecastToolSpec(api_key="test-key")
    output_path = tool.text_to_speech(
        text="Hello world",
        voice_id="voice1",
        output_path="output.mp3",
        model="ssfm-v21",
        language="eng",
        emotion_preset="happy",
        emotion_intensity=1.5,
        volume=120,
        audio_pitch=2,
        audio_tempo=1.2,
        audio_format="mp3",
        seed=42,
    )

    assert output_path == "output.mp3"
    call_args = mock_typecast.return_value.text_to_speech.call_args
    request = call_args[0][0]
    assert request.text == "Hello world"
    assert request.voice_id == "voice1"
    assert request.language == "eng"
    assert request.prompt.emotion_preset == "happy"
    assert request.prompt.emotion_intensity == 1.5
    assert request.output.volume == 120
    assert request.output.audio_pitch == 2
    assert request.output.audio_tempo == 1.2
    assert request.output.audio_format == "mp3"
    assert request.seed == 42
