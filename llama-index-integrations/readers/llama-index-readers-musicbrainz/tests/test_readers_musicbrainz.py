from unittest.mock import MagicMock, patch

import pytest

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.readers.musicbrainz import MusicBrainzReader


def test_class():
    names_of_base_classes = [b.__name__ for b in MusicBrainzReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


@pytest.fixture
def reader_with_mock_client():
    with patch(
        "llama_index.readers.musicbrainz.base.MusicBrainzReader.__init__",
        return_value=None,
    ):
        reader = MusicBrainzReader()
        reader._client = MagicMock()
        return reader


def test_search_artist_returns_documents(reader_with_mock_client):
    reader_with_mock_client._client.search_artists.return_value = {
        "artist-list": [
            {"name": "Radiohead", "country": "GB", "id": "abcd"},
            {"name": "Radiohead Tribute", "country": "US", "id": "efgh"},
        ]
    }
    docs = reader_with_mock_client.load_data(query="Radiohead", entity="artist")
    assert len(docs) == 2
    assert isinstance(docs[0], Document)
    assert docs[0].text == "Artist Radiohead (GB)"
    assert docs[0].metadata["entity"] == "artist"
    assert docs[0].metadata["id"] == "abcd"


def test_search_release_group_summary_uses_artist_credit_phrase(
    reader_with_mock_client,
):
    reader_with_mock_client._client.search_release_groups.return_value = {
        "release-group-list": [
            {
                "title": "OK Computer",
                "artist-credit-phrase": "Radiohead",
                "primary-type": "Album",
                "id": "ok-computer-mbid",
            }
        ]
    }
    docs = reader_with_mock_client.load_data(
        query="OK Computer", entity="release-group"
    )
    assert len(docs) == 1
    assert docs[0].text == "Album OK Computer by Radiohead"


def test_search_recording_falls_back_to_artist_credit_list(reader_with_mock_client):
    reader_with_mock_client._client.search_recordings.return_value = {
        "recording-list": [
            {
                "title": "Paranoid Android",
                "artist-credit": [{"artist": {"name": "Radiohead"}}],
                "id": "para-mbid",
            }
        ]
    }
    docs = reader_with_mock_client.load_data(
        query="Paranoid Android", entity="recording"
    )
    assert len(docs) == 1
    assert docs[0].text == "Recording Paranoid Android by Radiohead"


def test_lookup_by_mbid_returns_single_document(reader_with_mock_client):
    reader_with_mock_client._client.get_artist_by_id.return_value = {
        "artist": {
            "name": "Radiohead",
            "country": "GB",
            "id": "a74b1b7f-71a5-4011-9441-d0b5e4122711",
        }
    }
    docs = reader_with_mock_client.load_data(
        mbid="a74b1b7f-71a5-4011-9441-d0b5e4122711",
        entity="artist",
        includes=["release-groups"],
    )
    reader_with_mock_client._client.get_artist_by_id.assert_called_once_with(
        "a74b1b7f-71a5-4011-9441-d0b5e4122711", includes=["release-groups"]
    )
    assert len(docs) == 1
    assert docs[0].text == "Artist Radiohead (GB)"
    assert docs[0].metadata["id"] == "a74b1b7f-71a5-4011-9441-d0b5e4122711"


def test_lookup_returns_empty_when_payload_missing(reader_with_mock_client):
    """A 200-with-empty-body response should yield zero documents, not crash."""
    reader_with_mock_client._client.get_recording_by_id.return_value = {}
    docs = reader_with_mock_client.load_data(mbid="00000000-0000-0000-0000-000000000000", entity="recording")
    assert docs == []


def test_missing_query_and_mbid_raises(reader_with_mock_client):
    with pytest.raises(ValueError, match="query.+mbid"):
        reader_with_mock_client.load_data(entity="artist")


def test_unsupported_entity_raises(reader_with_mock_client):
    with pytest.raises(ValueError, match="Unsupported entity"):
        reader_with_mock_client.load_data(query="anything", entity="bogus")  # type: ignore[arg-type]
