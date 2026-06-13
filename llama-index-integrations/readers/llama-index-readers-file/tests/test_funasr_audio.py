from pathlib import Path
from unittest.mock import Mock, patch

from llama_index.readers.file import FunASRAudioReader


def test_funasr_audio_reader(tmp_path: Path) -> None:
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake audio")

    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "text": "Hello from FunASR",
    }

    with patch("requests.post", return_value=mock_response) as mock_post:
        reader = FunASRAudioReader(
            endpoint_url="http://localhost:8000",
            model="sensevoice",
        )
        docs = reader.load_data(audio_file)

    assert len(docs) == 1
    assert docs[0].text == "Hello from FunASR"
    assert docs[0].metadata["source"] == str(audio_file)
    assert docs[0].metadata["model"] == "sensevoice"
    assert docs[0].metadata["endpoint_url"] == "http://localhost:8000"
    assert docs[0].metadata["raw_response"] == {"text": "Hello from FunASR"}

    mock_post.assert_called_once()
