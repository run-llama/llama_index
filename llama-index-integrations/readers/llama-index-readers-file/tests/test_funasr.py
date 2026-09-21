from pathlib import Path
from unittest.mock import Mock

from llama_index.readers.file.funasr.base import FunASRReader


def test_funasr_reader_extracts_text(tmp_path: Path) -> None:
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake audio")

    recognizer = Mock()
    recognizer.generate.return_value = [
        {
            "key": "en",
            "text": "<|en|><|NEUTRAL|><|Speech|>hello from funasr",
        }
    ]

    reader = FunASRReader(
        model="iic/SenseVoiceSmall",
        device="cpu",
        recognizer=recognizer,
    )

    docs = reader.load_data(audio_file)

    assert len(docs) == 1
    assert docs[0].text == "hello from funasr"
    assert docs[0].metadata["source_path"] == str(audio_file)
    assert docs[0].metadata["model"] == "iic/SenseVoiceSmall"
    assert docs[0].metadata["device"] == "cpu"

    recognizer.generate.assert_called_once_with(input=str(audio_file))


def test_funasr_reader_keeps_tags_when_configured(tmp_path: Path) -> None:
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake audio")

    recognizer = Mock()
    recognizer.generate.return_value = [
        {
            "text": "<|en|><|NEUTRAL|><|Speech|>hello from funasr",
        }
    ]

    reader = FunASRReader(
        recognizer=recognizer,
        remove_tags=False,
    )

    docs = reader.load_data(audio_file)

    assert docs[0].text == "<|en|><|NEUTRAL|><|Speech|>hello from funasr"


def test_funasr_reader_extracts_segment_text(tmp_path: Path) -> None:
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake audio")

    recognizer = Mock()
    recognizer.generate.return_value = [
        {
            "segments": [
                {"text": "hello"},
                {"text": "from"},
                {"text": "segments"},
            ]
        }
    ]

    reader = FunASRReader(recognizer=recognizer)

    docs = reader.load_data(audio_file)

    assert docs[0].text == "hello from segments"


def test_funasr_reader_passes_generate_kwargs(tmp_path: Path) -> None:
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"fake audio")

    recognizer = Mock()
    recognizer.generate.return_value = [{"text": "hello"}]

    reader = FunASRReader(
        recognizer=recognizer,
        generate_kwargs={"batch_size_s": 60},
    )

    reader.load_data(audio_file)

    recognizer.generate.assert_called_once_with(
        input=str(audio_file),
        batch_size_s=60,
    )