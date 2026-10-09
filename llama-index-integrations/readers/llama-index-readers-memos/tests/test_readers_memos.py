from unittest.mock import patch

from llama_index.core.readers.base import BaseReader
from llama_index.readers.memos import MemosReader


def test_class():
    names_of_base_classes = [b.__name__ for b in MemosReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_load_data_uses_memo_all_endpoint_by_default():
    with patch("requests.get") as mock_get:
        mock_get.return_value.json.return_value = {"data": []}

        docs = MemosReader("https://example.com/").load_data()

        assert docs == []
        mock_get.assert_called_once_with("https://example.com/api/memo/all", {})


def test_load_data_sets_string_id_metadata_key():
    payload = {
        "data": [
            {
                "id": 7,
                "content": "hello",
                "creator": "me",
                "resourceList": [],
            }
        ]
    }

    with patch("requests.get") as mock_get:
        mock_get.return_value.json.return_value = payload

        docs = MemosReader("https://example.com/").load_data(params={"creator": "me"})

        assert len(docs) == 1
        assert docs[0].text == "hello"
        assert docs[0].metadata["id"] == 7
        assert docs[0].metadata["creator"] == "me"
        assert docs[0].metadata["resource_list"] == []
        mock_get.assert_called_once_with(
            "https://example.com/api/memo", {"creator": "me"}
        )
