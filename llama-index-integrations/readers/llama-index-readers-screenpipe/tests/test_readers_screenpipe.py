"""Tests for ScreenpipeReader."""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
import requests
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document
from llama_index.readers.screenpipe import ScreenpipeReader


@pytest.fixture()
def reader() -> ScreenpipeReader:
    return ScreenpipeReader(base_url="http://localhost:3030")


@pytest.fixture()
def mock_search_response() -> dict:
    return {
        "data": [
            {
                "type": "OCR",
                "content": {
                    "frame_id": 1,
                    "text": "Hello from the screen",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "file_path": "/tmp/frame.png",
                    "app_name": "Chrome",
                    "window_name": "Google",
                    "tags": [],
                    "browser_url": "https://google.com",
                },
            },
            {
                "type": "Audio",
                "content": {
                    "chunk_id": 1,
                    "transcription": "Meeting discussion about roadmap",
                    "timestamp": "2024-01-15T11:00:00Z",
                    "file_path": "/tmp/audio.wav",
                    "device_name": "MacBook Microphone",
                    "device_type": "Input",
                    "speaker": {"id": 1, "name": "John", "metadata": ""},
                },
            },
            {
                "type": "UI",
                "content": {
                    "id": 1,
                    "text": "Submit Button",
                    "timestamp": "2024-01-15T12:00:00Z",
                    "app_name": "VSCode",
                    "window_name": "main.py",
                },
            },
        ],
        "pagination": {"limit": 20, "offset": 0, "total": 3},
    }


def test_class():
    names_of_base_classes = [b.__name__ for b in ScreenpipeReader.__mro__]
    assert BasePydanticReader.__name__ in names_of_base_classes


def test_invalid_content_type(reader: ScreenpipeReader) -> None:
    with pytest.raises(ValueError, match="Invalid content_type"):
        reader.load_data(content_type="invalid")


@patch("llama_index.readers.screenpipe.base.requests.get")
def test_load_data(
    mock_get, reader: ScreenpipeReader, mock_search_response: dict
) -> None:
    mock_get.return_value.json.return_value = mock_search_response
    mock_get.return_value.raise_for_status.return_value = None

    documents = reader.load_data(content_type="all", limit=20)

    assert len(documents) == 3
    assert all(isinstance(doc, Document) for doc in documents)

    # Verify OCR document
    assert documents[0].text == "Hello from the screen"
    assert documents[0].metadata["type"] == "ocr"
    assert documents[0].metadata["app_name"] == "Chrome"
    assert documents[0].metadata["browser_url"] == "https://google.com"

    # Verify Audio document
    assert documents[1].text == "Meeting discussion about roadmap"
    assert documents[1].metadata["type"] == "audio"
    assert documents[1].metadata["speaker_name"] == "John"
    assert documents[1].metadata["device_name"] == "MacBook Microphone"

    # Verify UI document
    assert documents[2].text == "Submit Button"
    assert documents[2].metadata["type"] == "ui"
    assert documents[2].metadata["app_name"] == "VSCode"


@patch("llama_index.readers.screenpipe.base.requests.get")
def test_load_data_passes_params(mock_get, reader: ScreenpipeReader) -> None:
    mock_get.return_value.json.return_value = {"data": [], "pagination": {}}
    mock_get.return_value.raise_for_status.return_value = None

    reader.load_data(
        content_type="ocr",
        query="meeting",
        app_name="Zoom",
        window_name="Meeting",
        limit=10,
    )

    call_params = mock_get.call_args[1]["params"]
    assert call_params["content_type"] == "ocr"
    assert call_params["q"] == "meeting"
    assert call_params["app_name"] == "Zoom"
    assert call_params["window_name"] == "Meeting"
    assert call_params["limit"] == 10


@patch("llama_index.readers.screenpipe.base.requests.get")
def test_load_data_empty_response(mock_get, reader: ScreenpipeReader) -> None:
    mock_get.return_value.json.return_value = {"data": [], "pagination": {}}
    mock_get.return_value.raise_for_status.return_value = None

    documents = reader.load_data()
    assert documents == []


@patch("llama_index.readers.screenpipe.base.requests.get")
def test_unknown_item_type_skipped(mock_get, reader: ScreenpipeReader) -> None:
    mock_get.return_value.json.return_value = {
        "data": [{"type": "FutureType", "content": {"text": "something"}}],
        "pagination": {},
    }
    mock_get.return_value.raise_for_status.return_value = None

    documents = reader.load_data()
    assert documents == []


@patch("llama_index.readers.screenpipe.base.requests.get")
def test_naive_datetime_converted_to_utc(mock_get, reader: ScreenpipeReader) -> None:
    mock_get.return_value.json.return_value = {"data": [], "pagination": {}}
    mock_get.return_value.raise_for_status.return_value = None

    start = datetime(2024, 1, 15, 10, 0, 0)
    reader.load_data(start_time=start)

    call_params = mock_get.call_args[1]["params"]
    # Naive datetime is treated as local time and converted to UTC
    assert call_params["start_time"].endswith("Z")


@patch("llama_index.readers.screenpipe.base.requests.get")
def test_aware_datetime_serialized(mock_get, reader: ScreenpipeReader) -> None:
    mock_get.return_value.json.return_value = {"data": [], "pagination": {}}
    mock_get.return_value.raise_for_status.return_value = None

    start = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    reader.load_data(start_time=start)

    call_params = mock_get.call_args[1]["params"]
    assert call_params["start_time"] == "2024-01-15T10:00:00.000000Z"


@patch("llama_index.readers.screenpipe.base.requests.get")
def test_audio_without_speaker(mock_get, reader: ScreenpipeReader) -> None:
    mock_get.return_value.json.return_value = {
        "data": [
            {
                "type": "Audio",
                "content": {
                    "transcription": "Some audio",
                    "timestamp": "2024-01-15T11:00:00Z",
                    "device_name": "Mic",
                    "device_type": "Input",
                },
            },
        ],
        "pagination": {"limit": 20, "offset": 0, "total": 1},
    }
    mock_get.return_value.raise_for_status.return_value = None

    documents = reader.load_data()
    assert len(documents) == 1
    assert documents[0].text == "Some audio"
    assert "speaker_id" not in documents[0].metadata
    assert "speaker_name" not in documents[0].metadata


@patch("llama_index.readers.screenpipe.base.requests.get")
def test_pagination(mock_get, reader: ScreenpipeReader) -> None:
    page1 = {
        "data": [{"type": "OCR", "content": {"text": "page1", "timestamp": ""}}],
        "pagination": {"limit": 1, "offset": 0, "total": 2},
    }
    page2 = {
        "data": [{"type": "OCR", "content": {"text": "page2", "timestamp": ""}}],
        "pagination": {"limit": 1, "offset": 1, "total": 2},
    }
    mock_get.return_value.raise_for_status.return_value = None
    mock_get.return_value.json.side_effect = [page1, page2]

    documents = reader.load_data(limit=5)
    assert len(documents) == 2
    assert documents[0].text == "page1"
    assert documents[1].text == "page2"
    assert mock_get.call_count == 2


@patch("llama_index.readers.screenpipe.base.requests.get")
def test_http_error_raised(mock_get, reader: ScreenpipeReader) -> None:
    mock_get.return_value.raise_for_status.side_effect = (
        requests.exceptions.HTTPError("500 Server Error")
    )

    with pytest.raises(requests.exceptions.HTTPError):
        reader.load_data()


@patch("llama_index.readers.screenpipe.base.requests.get")
def test_trailing_slash_in_base_url(mock_get) -> None:
    mock_get.return_value.json.return_value = {"data": [], "pagination": {}}
    mock_get.return_value.raise_for_status.return_value = None

    reader = ScreenpipeReader(base_url="http://localhost:3030/")
    reader.load_data()

    called_url = mock_get.call_args[0][0]
    assert called_url == "http://localhost:3030/search"
