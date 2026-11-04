from types import SimpleNamespace

import pytest

from llama_index.core.readers.base import BaseReader
from llama_index.readers.youtube_transcript import YoutubeTranscriptReader
from llama_index.readers.youtube_transcript import base as reader_module


def test_class():
    names_of_base_classes = [b.__name__ for b in YoutubeTranscriptReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


class FakeTranscriptApi:
    """Stands in for youtube_transcript_api.YouTubeTranscriptApi (1.x instance API)."""

    calls = []
    transcripts = {}

    def fetch(self, video_id, languages=("en",)):
        FakeTranscriptApi.calls.append((video_id, list(languages)))
        if video_id not in FakeTranscriptApi.transcripts:
            raise RuntimeError(f"no transcript for {video_id}")
        return [
            SimpleNamespace(text=text, start=float(i), duration=1.0)
            for i, text in enumerate(FakeTranscriptApi.transcripts[video_id])
        ]


@pytest.fixture
def fake_api(monkeypatch):
    FakeTranscriptApi.calls = []
    FakeTranscriptApi.transcripts = {
        "aircAruvnKk": ["first line", "second line"],
        "jNQXAC9IVRw": ["only line"],
    }
    monkeypatch.setattr(reader_module, "YouTubeTranscriptApi", FakeTranscriptApi)
    return FakeTranscriptApi


def test_load_data_joins_snippet_text(fake_api):
    docs = YoutubeTranscriptReader().load_data(
        ["https://www.youtube.com/watch?v=aircAruvnKk"]
    )

    assert len(docs) == 1
    assert docs[0].text == "first line\nsecond line"
    assert docs[0].id_ == "aircAruvnKk"
    assert docs[0].metadata == {"video_id": "aircAruvnKk"}


def test_load_data_defaults_to_english(fake_api):
    YoutubeTranscriptReader().load_data(["https://youtu.be/jNQXAC9IVRw"])

    assert fake_api.calls == [("jNQXAC9IVRw", ["en"])]


def test_load_data_passes_languages(fake_api):
    YoutubeTranscriptReader().load_data(
        ["https://www.youtube.com/embed/aircAruvnKk"], languages=["de", "en"]
    )

    assert fake_api.calls == [("aircAruvnKk", ["de", "en"])]


def test_load_data_multiple_links(fake_api):
    docs = YoutubeTranscriptReader().load_data(
        [
            "https://www.youtube.com/watch?v=aircAruvnKk",
            "https://youtu.be/jNQXAC9IVRw",
        ]
    )

    assert [doc.id_ for doc in docs] == ["aircAruvnKk", "jNQXAC9IVRw"]
    assert docs[1].text == "only line"


def test_unsupported_url_raises_value_error(fake_api):
    with pytest.raises(ValueError, match="not a supported youtube URL"):
        YoutubeTranscriptReader().load_data(["https://example.com/watch?v=abc"])

    assert fake_api.calls == []


def test_fetch_errors_are_not_swallowed(fake_api):
    with pytest.raises(RuntimeError, match="no transcript for missing"):
        YoutubeTranscriptReader().load_data(["https://www.youtube.com/watch?v=missing"])
