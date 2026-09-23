from pathlib import Path
from unittest.mock import Mock, patch

from llama_index.readers.file import FunASRAudioReader


def test_funasr_audio_reader_text_response(tmp_path: Path) -> None:
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake audio")

    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"text": "Hello from FunASR"}

    with patch("requests.post", return_value=mock_response) as mock_post:
        reader = FunASRAudioReader(
            endpoint_url="http://localhost:8000",
            model="sensevoice",
            language="en",
        )
        docs = reader.load_data(audio_file)

    assert len(docs) == 1
    assert docs[0].text == "Hello from FunASR"
    assert docs[0].metadata["source_path"] == str(audio_file)
    assert docs[0].metadata["model"] == "sensevoice"
    assert docs[0].metadata["endpoint"] == (
        "http://localhost:8000/v1/audio/transcriptions"
    )

    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert kwargs["data"]["model"] == "sensevoice"
    assert kwargs["data"]["language"] == "en"


def test_funasr_audio_reader_transcript_response(tmp_path: Path) -> None:
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake audio")

    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"transcript": "Hello transcript"}

    with patch("requests.post", return_value=mock_response):
        reader = FunASRAudioReader()
        docs = reader.load_data(audio_file)

    assert docs[0].text == "Hello transcript"


def test_funasr_audio_reader_transcription_response(tmp_path: Path) -> None:
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake audio")

    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"transcription": "Hello transcription"}

    with patch("requests.post", return_value=mock_response):
        reader = FunASRAudioReader()
        docs = reader.load_data(audio_file)

    assert docs[0].text == "Hello transcription"


def test_funasr_audio_reader_segments_response(tmp_path: Path) -> None:
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake audio")

    segments = [
        {"text": "Hello"},
        {"text": "from"},
        {"text": "segments"},
    ]

    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"segments": segments}

    with patch("requests.post", return_value=mock_response):
        reader = FunASRAudioReader()
        docs = reader.load_data(audio_file)

    assert docs[0].text == "Hello from segments"
    assert docs[0].metadata["segments"] == segments