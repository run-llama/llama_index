from unittest.mock import patch

import pytest
from llama_index.readers.mongo import SimpleMongoReader
from llama_index.schema import MetadataMode

try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None  # type: ignore


@pytest.mark.skipif(MongoClient is None, reason="pymongo not installed")
def test_load_data() -> None:
    """Test Mongo reader using default field_names."""
    mock_cursor = [{"text": "one"}, {"text": "two"}, {"text": "three"}]

    with patch("pymongo.collection.Collection.find") as mock_find:
        mock_find.return_value = mock_cursor

        reader = SimpleMongoReader("host", 1)
        documents = reader.load_data("my_db", "my_collection")

        assert len(documents) == 3
        assert documents[0].get_content() == "one"
        assert documents[1].get_content() == "two"
        assert documents[2].get_content() == "three"


@pytest.mark.skipif(MongoClient is None, reason="pymongo not installed")
def test_load_data_with_field_name() -> None:
    """Test Mongo reader using passed in field_names."""
    mock_cursor = [
        {"first": "first1", "second": ["second1", "second11"], "third": "third1"},
        {"first": "first2", "second": ["second2", "second22"], "third": "third2"},
        {"first": "first3", "second": ["second3", "second33"], "third": "third3"},
    ]

    with patch("pymongo.collection.Collection.find") as mock_find:
        mock_find.return_value = mock_cursor

        reader = SimpleMongoReader("host", 1)
        documents = reader.load_data(
            "my_db", "my_collection", field_names=["first", "second", "third"]
        )

        assert len(documents) == 3
        assert documents[0].get_content() == "first1second1second11third1"
        assert documents[1].get_content() == "first2second2second22third2"
        assert documents[2].get_content() == "first3second3second33third3"


@pytest.mark.skipif(MongoClient is None, reason="pymongo not installed")
def test_load_data_with_metadata_name() -> None:
    """Test Mongo reader using passed in metadata_name."""
    mock_cursor = [
        {"first": "first1", "second": "second1", "third": "third1"},
        {"first": "first2", "second": "second2", "third": "third2"},
        {"first": "first3", "second": "second3", "third": "third3"},
    ]

    with patch("pymongo.collection.Collection.find") as mock_find:
        mock_find.return_value = mock_cursor

        reader = SimpleMongoReader("host", 1)
        documents = reader.load_data(
            "my_db",
            "my_collection",
            field_names=["first"],
            metadata_names=["second", "third"],
        )

        assert len(documents) == 3
        assert documents[0].get_metadata_str() == "second: second1\nthird: third1"
        assert documents[1].get_metadata_str() == "second: second2\nthird: third2"
        assert documents[2].get_metadata_str() == "second: second3\nthird: third3"
        assert (
            documents[0].get_content(metadata_mode=MetadataMode.ALL)
            == "second: second1\nthird: third1\n\nfirst1"
        )
        assert (
            documents[1].get_content(metadata_mode=MetadataMode.ALL)
            == "second: second2\nthird: third2\n\nfirst2"
        )
        assert (
            documents[2].get_content(metadata_mode=MetadataMode.ALL)
            == "second: second3\nthird: third3\n\nfirst3"
        )
