from unittest.mock import patch

import pytest

from llama_index.readers.mongo import SimpleMongoReader

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
        assert documents[0].text == "one"
        assert documents[1].text == "two"
        assert documents[2].text == "three"


@pytest.mark.skipif(MongoClient is None, reason="pymongo not installed")
def test_load_data_with_field_name() -> None:
    """Test Mongo reader using passed in field_names."""
    mock_cursor = [
        {"first": "first1", "second": "second1", "third": "third1"},
        {"first": "first2", "second": "second2", "third": "third2"},
        {"first": "first3", "second": "second3", "third": "third3"},
    ]

    with patch("pymongo.collection.Collection.find") as mock_find:
        mock_find.return_value = mock_cursor

        reader = SimpleMongoReader("host", 1)
        documents = reader.load_data(
            "my_db", "my_collection", field_names=["first", "second", "third"]
        )

        assert len(documents) == 3
        assert documents[0].text == "first1second1third1"
        assert documents[1].text == "first2second2third2"
        assert documents[2].text == "first3second3third3"
