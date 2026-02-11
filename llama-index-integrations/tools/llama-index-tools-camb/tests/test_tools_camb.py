"""Tests for CAMB AI tool spec."""

import json
from unittest.mock import MagicMock, Mock, patch, mock_open

import pytest
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.camb import CambToolSpec


# ---------------------------------------------------------------------------
# Class structure
# ---------------------------------------------------------------------------


def test_class_inherits_base_tool_spec():
    names_of_base_classes = [b.__name__ for b in CambToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_spec_functions():
    assert "text_to_speech" in CambToolSpec.spec_functions
    assert "translate" in CambToolSpec.spec_functions
    assert "transcribe" in CambToolSpec.spec_functions
    assert "translated_tts" in CambToolSpec.spec_functions
    assert "clone_voice" in CambToolSpec.spec_functions
    assert "list_voices" in CambToolSpec.spec_functions
    assert "text_to_sound" in CambToolSpec.spec_functions
    assert "separate_audio" in CambToolSpec.spec_functions


def test_spec_functions_count():
    assert len(CambToolSpec.spec_functions) == 9


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


@patch("camb.client.CambAI")
def test_init_with_api_key(mock_client):
    tool = CambToolSpec(api_key="test-key")
    assert tool.api_key == "test-key"
    mock_client.assert_called_once()


@patch("camb.client.CambAI")
def test_init_with_env_var(mock_client):
    with patch.dict("os.environ", {"CAMB_API_KEY": "env-key"}):
        tool = CambToolSpec()
        assert tool.api_key == "env-key"


def test_init_without_key_raises():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="CAMB AI API key is required"):
            CambToolSpec()


# ---------------------------------------------------------------------------
# text_to_speech
# ---------------------------------------------------------------------------


@patch("camb.client.CambAI")
def test_text_to_speech(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.text_to_speech.tts.return_value = [b"audio_chunk_1", b"audio_chunk_2"]

    tool = CambToolSpec(api_key="test-key")
    result = tool.text_to_speech("Hello world")

    assert result.endswith(".wav")
    mock_client.text_to_speech.tts.assert_called_once()


# ---------------------------------------------------------------------------
# translate
# ---------------------------------------------------------------------------


@patch("camb.client.CambAI")
def test_translate(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_result = Mock()
    mock_result.text = "Hola mundo"
    mock_client.translation.translation_stream.return_value = mock_result

    tool = CambToolSpec(api_key="test-key")
    result = tool.translate("Hello world", source_language=1, target_language=2)

    assert result == "Hola mundo"


@patch("camb.client.CambAI")
def test_translate_api_error_workaround(mock_client_cls):
    """Test that ApiError with status 200 is handled as success."""
    from camb.core.api_error import ApiError

    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    error = ApiError(status_code=200, body="Hola mundo")
    mock_client.translation.translation_stream.side_effect = error

    tool = CambToolSpec(api_key="test-key")
    result = tool.translate("Hello world", source_language=1, target_language=2)

    assert result == "Hola mundo"


# ---------------------------------------------------------------------------
# list_voices
# ---------------------------------------------------------------------------


@patch("camb.client.CambAI")
def test_list_voices(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_voice = Mock()
    mock_voice.id = 123
    mock_voice.voice_name = "Test Voice"
    mock_voice.gender = 1
    mock_voice.age = 30
    mock_voice.language = 1
    mock_client.voice_cloning.list_voices.return_value = [mock_voice]

    tool = CambToolSpec(api_key="test-key")
    result = tool.list_voices()

    voices = json.loads(result)
    assert len(voices) == 1
    assert voices[0]["id"] == 123
    assert voices[0]["name"] == "Test Voice"
    assert voices[0]["gender"] == "male"


# ---------------------------------------------------------------------------
# clone_voice
# ---------------------------------------------------------------------------


@patch("camb.client.CambAI")
def test_clone_voice(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_result = Mock()
    mock_result.voice_id = 999
    mock_result.message = "Voice created"
    mock_client.voice_cloning.create_custom_voice.return_value = mock_result

    tool = CambToolSpec(api_key="test-key")

    with patch("builtins.open", mock_open(read_data=b"audio_data")):
        result = tool.clone_voice(
            voice_name="My Voice",
            audio_file_path="/fake/path.wav",
            gender=1,
        )

    out = json.loads(result)
    assert out["voice_id"] == 999
    assert out["status"] == "created"


# ---------------------------------------------------------------------------
# transcribe
# ---------------------------------------------------------------------------


@patch("httpx.get")
@patch("camb.client.CambAI")
def test_transcribe(mock_client_cls, mock_httpx_get):
    mock_resp = Mock()
    mock_resp.content = b"fake audio data"
    mock_resp.raise_for_status = Mock()
    mock_httpx_get.return_value = mock_resp

    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_create = Mock()
    mock_create.task_id = "task-123"
    mock_client.transcription.create_transcription.return_value = mock_create

    mock_status = Mock()
    mock_status.status = "completed"
    mock_status.run_id = "run-456"
    mock_client.transcription.get_transcription_task_status.return_value = mock_status

    mock_transcription = Mock()
    mock_transcription.text = "Hello world"
    mock_transcription.segments = []
    mock_transcription.speakers = []
    mock_client.transcription.get_transcription_result.return_value = mock_transcription

    tool = CambToolSpec(api_key="test-key")
    result = tool.transcribe(language=1, audio_url="https://example.com/audio.mp3")

    out = json.loads(result)
    assert out["text"] == "Hello world"


def test_transcribe_no_source():
    with patch("camb.client.CambAI"):
        tool = CambToolSpec(api_key="test-key")
        result = tool.transcribe(language=1)
        out = json.loads(result)
        assert "error" in out


# ---------------------------------------------------------------------------
# text_to_sound
# ---------------------------------------------------------------------------


@patch("camb.client.CambAI")
def test_text_to_sound(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_create = Mock()
    mock_create.task_id = "task-789"
    mock_client.text_to_audio.create_text_to_audio.return_value = mock_create

    mock_status = Mock()
    mock_status.status = "completed"
    mock_status.run_id = "run-012"
    mock_client.text_to_audio.get_text_to_audio_status.return_value = mock_status

    mock_client.text_to_audio.get_text_to_audio_result.return_value = [
        b"chunk1", b"chunk2"
    ]

    tool = CambToolSpec(api_key="test-key")
    result = tool.text_to_sound("upbeat electronic music")

    assert result.endswith(".wav")


# ---------------------------------------------------------------------------
# to_tool_list
# ---------------------------------------------------------------------------


@patch("camb.client.CambAI")
def test_to_tool_list(mock_client_cls):
    tool = CambToolSpec(api_key="test-key")
    tools = tool.to_tool_list()
    assert len(tools) == 9
    tool_names = {t.metadata.name for t in tools}
    assert "text_to_speech" in tool_names
    assert "translate" in tool_names
    assert "separate_audio" in tool_names


# ---------------------------------------------------------------------------
# Helper: audio format detection
# ---------------------------------------------------------------------------


def test_detect_audio_format_wav():
    assert CambToolSpec._detect_audio_format(b"RIFF" + b"\x00" * 100) == "wav"


def test_detect_audio_format_mp3():
    assert CambToolSpec._detect_audio_format(b"\xff\xfb" + b"\x00" * 100) == "mp3"


def test_detect_audio_format_flac():
    assert CambToolSpec._detect_audio_format(b"fLaC" + b"\x00" * 100) == "flac"


def test_detect_audio_format_ogg():
    assert CambToolSpec._detect_audio_format(b"OggS" + b"\x00" * 100) == "ogg"


def test_detect_audio_format_content_type():
    assert CambToolSpec._detect_audio_format(b"\x00" * 100, "audio/mpeg") == "mp3"


def test_detect_audio_format_unknown():
    assert CambToolSpec._detect_audio_format(b"\x00" * 100) == "pcm"


# ---------------------------------------------------------------------------
# Helper: WAV header
# ---------------------------------------------------------------------------


def test_add_wav_header():
    pcm = b"\x00" * 100
    wav = CambToolSpec._add_wav_header(pcm)
    assert wav.startswith(b"RIFF")
    assert b"WAVE" in wav[:12]
    assert wav.endswith(pcm)
