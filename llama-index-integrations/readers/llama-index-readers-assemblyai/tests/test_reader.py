from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import assemblyai as aai
from llama_index.core.readers.base import BaseReader
from llama_index.readers.assemblyai import (
    AssemblyAIAudioTranscriptReader,
    TranscriptFormat,
)


def test_class_hierarchy():
    assert issubclass(AssemblyAIAudioTranscriptReader, BaseReader)


def test_init_validation():
    # Neither provided
    with pytest.raises(
        ValueError, match="Either file_path or transcript_id must be provided"
    ):
        AssemblyAIAudioTranscriptReader()

    # Both provided
    with pytest.raises(
        ValueError, match="Only one of file_path or transcript_id should be provided"
    ):
        AssemblyAIAudioTranscriptReader(file_path="audio.mp3", transcript_id="tr_123")

    # API key setting
    with patch.object(aai.settings, "api_key", None):
        AssemblyAIAudioTranscriptReader(file_path="audio.mp3", api_key="test-api-key")
        assert aai.settings.api_key == "test-api-key"


def _create_mock_transcript(**kwargs: Any) -> MagicMock:
    transcript = MagicMock()
    transcript.error = None
    transcript.text = kwargs.get("text", "This is a transcribed sentence.")
    transcript.json_response = kwargs.get(
        "json_response",
        {"id": "tr_1", "status": "completed", "confidence": 0.98},
    )
    transcript.export_subtitles_srt.return_value = kwargs.get(
        "subtitles_srt", "1\n00:00:00,000 --> 00:00:02,000\nHello world\n"
    )
    transcript.export_subtitles_vtt.return_value = kwargs.get(
        "subtitles_vtt", "WEBVTT\n\n1\n00:00:00.000 --> 00:00:02.000\nHello world\n"
    )

    sentence = MagicMock()
    sentence.text = "This is a sentence."
    sentence.dict.return_value = {"start": 0, "end": 1000}
    sentence.model_dump.return_value = {"start": 0, "end": 1000}
    transcript.get_sentences.return_value = kwargs.get("sentences", [sentence])

    paragraph = MagicMock()
    paragraph.text = "This is a paragraph."
    paragraph.dict.return_value = {"start": 0, "end": 3000}
    paragraph.model_dump.return_value = {"start": 0, "end": 3000}
    transcript.get_paragraphs.return_value = kwargs.get("paragraphs", [paragraph])

    chapter = MagicMock()
    chapter.summary = "Overview of the discussion."
    chapter.headline = "Introduction"
    chapter.gist = "Greeting"
    chapter.dict.return_value = {
        "headline": "Introduction",
        "gist": "Greeting",
        "start": 0,
        "end": 5000,
    }
    chapter.model_dump.return_value = {
        "headline": "Introduction",
        "gist": "Greeting",
        "start": 0,
        "end": 5000,
    }
    transcript.chapters = kwargs.get("chapters", [chapter])

    utterance = MagicMock()
    utterance.text = "Speaker utterance text."
    utterance.speaker = "A"
    utterance.dict.return_value = {"speaker": "A", "start": 0, "end": 1500}
    utterance.model_dump.return_value = {"speaker": "A", "start": 0, "end": 1500}
    transcript.utterances = kwargs.get("utterances", [utterance])

    return transcript


def test_load_data_text():
    mock_transcript = _create_mock_transcript(text="Hello world")
    with patch.object(
        aai.Transcriber, "transcribe", return_value=mock_transcript
    ) as mock_transcribe:
        reader = AssemblyAIAudioTranscriptReader(
            file_path="audio.mp3", transcript_format=TranscriptFormat.TEXT
        )
        docs = reader.load_data(extra_info={"source": "test_audio"})

        assert len(docs) == 1
        assert docs[0].text == "Hello world"
        assert docs[0].metadata["id"] == "tr_1"
        assert docs[0].metadata["source"] == "test_audio"
        mock_transcribe.assert_called_once_with("audio.mp3")


def test_load_data_with_override_file_path():
    mock_transcript = _create_mock_transcript(text="Loaded via argument")
    with patch.object(
        aai.Transcriber, "transcribe", return_value=mock_transcript
    ) as mock_transcribe:
        reader = AssemblyAIAudioTranscriptReader(file_path="default.mp3")
        docs = reader.load_data("custom.mp3", extra_info={"tag": "custom"})

        assert len(docs) == 1
        assert docs[0].text == "Loaded via argument"
        assert docs[0].metadata["tag"] == "custom"
        mock_transcribe.assert_called_once_with("custom.mp3")


def test_load_data_sentences():
    mock_transcript = _create_mock_transcript()
    with patch.object(aai.Transcriber, "transcribe", return_value=mock_transcript):
        reader = AssemblyAIAudioTranscriptReader(
            file_path="audio.mp3", transcript_format=TranscriptFormat.SENTENCES
        )
        docs = reader.load_data(extra_info={"filename": "sample.mp3"})

        assert len(docs) == 1
        assert docs[0].text == "This is a sentence."
        assert docs[0].metadata["start"] == 0
        assert docs[0].metadata["end"] == 1000
        assert docs[0].metadata["filename"] == "sample.mp3"


def test_load_data_paragraphs():
    mock_transcript = _create_mock_transcript()
    with patch.object(aai.Transcriber, "transcribe", return_value=mock_transcript):
        reader = AssemblyAIAudioTranscriptReader(
            file_path="audio.mp3", transcript_format=TranscriptFormat.PARAGRAPHS
        )
        docs = reader.load_data()

        assert len(docs) == 1
        assert docs[0].text == "This is a paragraph."
        assert docs[0].metadata["start"] == 0
        assert docs[0].metadata["end"] == 3000


def test_load_data_chapters():
    mock_transcript = _create_mock_transcript()
    with patch.object(aai.Transcriber, "transcribe", return_value=mock_transcript):
        reader = AssemblyAIAudioTranscriptReader(
            file_path="audio.mp3", transcript_format=TranscriptFormat.CHAPTERS
        )
        docs = reader.load_data(extra_info={"domain": "finance"})

        assert len(docs) == 1
        assert docs[0].text == "Overview of the discussion."
        assert docs[0].metadata["headline"] == "Introduction"
        assert docs[0].metadata["gist"] == "Greeting"
        assert docs[0].metadata["domain"] == "finance"


def test_load_data_utterances():
    mock_transcript = _create_mock_transcript()
    with patch.object(aai.Transcriber, "transcribe", return_value=mock_transcript):
        reader = AssemblyAIAudioTranscriptReader(
            file_path="audio.mp3", transcript_format=TranscriptFormat.UTTERANCES
        )
        docs = reader.load_data()

        assert len(docs) == 1
        assert docs[0].text == "Speaker utterance text."
        assert docs[0].metadata["speaker"] == "A"
        assert docs[0].metadata["start"] == 0
        assert docs[0].metadata["end"] == 1500


def test_load_data_subtitles_srt_and_vtt():
    mock_transcript = _create_mock_transcript()
    with patch.object(aai.Transcriber, "transcribe", return_value=mock_transcript):
        reader_srt = AssemblyAIAudioTranscriptReader(
            file_path="audio.mp3", transcript_format=TranscriptFormat.SUBTITLES_SRT
        )
        docs_srt = reader_srt.load_data(extra_info={"format": "srt"})
        assert len(docs_srt) == 1
        assert "00:00:00,000" in docs_srt[0].text
        assert docs_srt[0].metadata["id"] == "tr_1"
        assert docs_srt[0].metadata["format"] == "srt"

        reader_vtt = AssemblyAIAudioTranscriptReader(
            file_path="audio.mp3", transcript_format=TranscriptFormat.SUBTITLES_VTT
        )
        docs_vtt = reader_vtt.load_data(extra_info={"format": "vtt"})
        assert len(docs_vtt) == 1
        assert "WEBVTT" in docs_vtt[0].text
        assert docs_vtt[0].metadata["id"] == "tr_1"
        assert docs_vtt[0].metadata["format"] == "vtt"


def test_load_data_by_transcript_id():
    mock_transcript = _create_mock_transcript(text="Loaded by ID")
    with patch.object(
        aai.Transcript, "get_by_id", return_value=mock_transcript
    ) as mock_get_by_id:
        reader = AssemblyAIAudioTranscriptReader(transcript_id="tr_existing_123")
        docs = reader.load_data()

        assert len(docs) == 1
        assert docs[0].text == "Loaded by ID"
        mock_get_by_id.assert_called_once_with("tr_existing_123")


def test_lazy_load_data():
    mock_transcript = _create_mock_transcript(text="Lazy text")
    with patch.object(aai.Transcriber, "transcribe", return_value=mock_transcript):
        reader = AssemblyAIAudioTranscriptReader(file_path="audio.mp3")
        gen = reader.lazy_load_data()
        docs = list(gen)
        assert len(docs) == 1
        assert docs[0].text == "Lazy text"


def test_error_handling():
    mock_transcript = MagicMock()
    mock_transcript.error = "File format not supported"
    with patch.object(aai.Transcriber, "transcribe", return_value=mock_transcript):
        reader = AssemblyAIAudioTranscriptReader(file_path="bad.xyz")
        with pytest.raises(
            ValueError, match="Could not transcribe file: File format not supported"
        ):
            reader.load_data()


def test_empty_transcript_text():
    mock_transcript = _create_mock_transcript(text=None)
    with patch.object(aai.Transcriber, "transcribe", return_value=mock_transcript):
        reader = AssemblyAIAudioTranscriptReader(file_path="silent.mp3")
        docs = reader.load_data()
        assert len(docs) == 1
        assert docs[0].text == ""
