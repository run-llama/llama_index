from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.readers.base import BaseReader
from llama_index.readers.twelvelabs import TwelveLabsVideoReader
from llama_index.readers.twelvelabs.base import _extract_text


def test_class():
    names_of_base_classes = [b.__name__ for b in TwelveLabsVideoReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("TWELVELABS_API_KEY", raising=False)
    with pytest.raises(ValueError):
        TwelveLabsVideoReader()


def test_reads_key_from_env(monkeypatch):
    monkeypatch.setenv("TWELVELABS_API_KEY", "tlk_env")
    reader = TwelveLabsVideoReader()
    assert reader._api_key == "tlk_env"


def test_load_data_requires_exactly_one_source():
    reader = TwelveLabsVideoReader(api_key="tlk_test")
    with pytest.raises(ValueError):
        reader.load_data()
    with pytest.raises(ValueError):
        reader.load_data(video_url="https://x/v.mp4", asset_id="a1")


def _resp(json_body, ok=True, status=200):
    response = MagicMock()
    response.ok = ok
    response.status_code = status
    response.json.return_value = json_body
    response.text = str(json_body)
    return response


def test_load_data_with_asset_id_returns_document():
    reader = TwelveLabsVideoReader(api_key="tlk_test", model="pegasus1.5")
    with patch("llama_index.readers.twelvelabs.base.requests") as req:
        req.post.return_value = _resp({"task_id": "task_1", "status": "pending"})
        req.get.return_value = _resp(
            {
                "status": "ready",
                "result": {"data": "A demo of a checkout bug at [0:05]."},
            }
        )
        docs = reader.load_data(asset_id="asset_1")

    assert len(docs) == 1
    assert "checkout bug" in docs[0].text
    assert docs[0].metadata["asset_id"] == "asset_1"
    assert docs[0].metadata["task_id"] == "task_1"
    assert docs[0].metadata["provider"] == "twelvelabs"
    # asset_id path does no upload — only the analyze POST happens.
    assert req.post.call_count == 1


def test_load_data_with_url_uploads_then_analyzes():
    reader = TwelveLabsVideoReader(api_key="tlk_test")
    with patch("llama_index.readers.twelvelabs.base.requests") as req:
        req.post.side_effect = [
            _resp({"_id": "asset_9", "status": "ready"}),  # POST /assets (url)
            _resp({"task_id": "task_9", "status": "pending"}),  # POST /analyze/tasks
        ]
        req.get.return_value = _resp({"status": "ready", "result": {"data": "ok text"}})
        docs = reader.load_data(video_url="https://example.com/clip.mp4")

    assert docs[0].text == "ok text"
    assert docs[0].metadata["asset_id"] == "asset_9"
    assert req.post.call_count == 2  # asset register + analyze


def test_analyze_failure_raises():
    reader = TwelveLabsVideoReader(api_key="tlk_test")
    with patch("llama_index.readers.twelvelabs.base.requests") as req:
        req.post.return_value = _resp({"task_id": "t", "status": "pending"})
        req.get.return_value = _resp({"status": "failed", "result": None})
        with pytest.raises(RuntimeError):
            reader.load_data(asset_id="asset_1")


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ("plain text", "plain text"),
        ({"data": "the data"}, "the data"),
        ({"text": "the text"}, "the text"),
        ({"unknown": 1}, ""),
        (None, ""),
    ],
)
def test_extract_text(result, expected):
    assert _extract_text(result) == expected
