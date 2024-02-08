from unittest.mock import MagicMock, patch

import pytest
from llama_index.readers.arango_db import SimpleArangoDBReader


@pytest.fixture()
def mock_arangodb_client():
    with patch("arango.ArangoClient") as mock_client:
        # Mock the behavior of the db and collection
        mock_db = MagicMock()
        mock_students_collection = MagicMock()

        mock_students = [
            {"_key": "1", "name": "Alice", "age": 20},
            {"_key": "2", "name": "Bob", "age": 21},
            {"_key": "3", "name": "Mark", "age": 20},
        ]

        mock_students_collection.find.return_value = mock_students
        mock_db.collection.return_value = mock_students_collection
        mock_client.db.return_value = mock_db

        yield mock_client


def test_load_students(mock_arangodb_client):
    reader = SimpleArangoDBReader(client=mock_arangodb_client)
    documents = reader.load_data(
        username="usr",
        password="pass",
        db_name="school",
        collection_name="students",
        field_names=["name", "age"],
    )

    assert len(documents) == 3
    assert documents[0].text == "Alice 20"
    assert documents[1].text == "Bob 21"
    assert documents[2].text == "Mark 20"
