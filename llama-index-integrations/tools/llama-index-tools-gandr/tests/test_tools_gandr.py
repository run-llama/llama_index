from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.gandr import GandrToolSpec
from llama_index.tools.gandr.base import GANDR_VOICES, MAX_INPUT_CHARACTERS


def test_class():
    names_of_base_classes = [b.__name__ for b in GandrToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_get_voices():
    tool = GandrToolSpec(api_key="gnd_test")
    assert tool.get_voices() == list(GANDR_VOICES)


def test_text_to_speech(tmp_path):
    tool = GandrToolSpec(api_key="gnd_test")

    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"audio-bytes"]
    mock_response.raise_for_status.return_value = None

    output_path = tmp_path / "speech.mp3"
    with patch("requests.post", return_value=mock_response) as mock_post:
        result = tool.text_to_speech("Hello world", str(output_path))

    assert result == str(output_path)
    assert output_path.read_bytes() == b"audio-bytes"

    args, kwargs = mock_post.call_args
    assert args[0] == "https://tts.gandr.ai/v1/audio/speech"
    assert kwargs["headers"]["Authorization"] == "Bearer gnd_test"
    assert kwargs["json"] == {
        "model": "tts-1",
        "input": "Hello world",
        "voice": "gandr-mia",
        "response_format": "mp3",
    }


def test_text_to_speech_input_cap(tmp_path):
    tool = GandrToolSpec(api_key="gnd_test")
    too_long = "a" * (MAX_INPUT_CHARACTERS + 1)
    with pytest.raises(ValueError):
        tool.text_to_speech(too_long, str(tmp_path / "speech.mp3"))
