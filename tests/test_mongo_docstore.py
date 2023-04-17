from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch
import uuid

import pytest

from gpt_index.docstore.mongo_docstore import MongoDocumentStore
from gpt_index.readers.schema.base import Document
from gpt_index.schema import BaseDocument


class MockMongoCollection:
    def __init__(self) -> None:
        self._data: Dict[str, dict] = {}

    def find_one(self, filter: dict) -> Optional[dict]:
        for data in self._data.values():
            if filter is None or all(data[key] == val for key, val in filter.items()):
                return data.copy()
        return None

    def find(self, filter: Optional[dict] = None) -> List[dict]:
        data_list = []
        for data in self._data.values():
            if filter is None or all(data[key] == val for key, val in filter.items()):
                data_list.append(data.copy())
        return data_list

    def delete_one(self, filter: dict) -> Any:
        matched = self.find_one(filter)
        if matched is not None:
            del self._data[matched["_id"]]

        delete_result = Mock()
        delete_result.deleted_count = 1 if matched else 0
        return delete_result

    def replace_one(self, filter: dict, obj: dict, upsert: bool = False) -> Any:
        matched = self.find_one(filter)
        if matched is not None:
            self.insert_one(obj, matched["_id"])
        elif upsert:
            self.insert_one(obj)

        update_result = Mock()
        return update_result

    def insert_one(self, obj: dict, _id: Optional[str] = None) -> Any:
        _id = _id or str(uuid.uuid4())
        obj["_id"] = _id
        self._data[_id] = obj

        insert_result = Mock()
        insert_result.inserted_id = _id
        return insert_result

    def update_one(self, filter: dict, update: dict, upsert: bool = False) -> Any:
        matched = self.find_one(filter)
        if matched is not None:
            _id = matched["_id"]
            self._data[_id].update(update)
        else:
            if upsert:
                self.insert_one(update)

    def insert_many(self, objs: List[dict]) -> Any:
        results = [self.insert_one(obj) for obj in objs]
        inserted_ids = [result.inserted_id for result in results]

        insert_result = Mock()
        insert_result.inserted_ids = inserted_ids
        return insert_result


class MockMongoDB:
    def __init__(self) -> None:
        self._collection = MockMongoCollection()
        self._hash_collection = MockMongoCollection()

    def __getitem__(self, collection: str) -> MockMongoCollection:
        if collection.endswith("hash"):
            return self._hash_collection
        else:
            return self._collection


class MockMongoClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._db = MockMongoDB()

    def __getitem__(self, db: str) -> MockMongoDB:
        del db
        return self._db


@pytest.fixture
def documents() -> List[Document]:
    return [
        Document("doc_1"),
        Document("doc_2"),
    ]


def test_mongo_docstore(documents: List[Document]) -> None:
    ds = MongoDocumentStore(mongo_client=MockMongoClient())  # type: ignore
    assert len(ds.docs) == 0

    # test adding documents
    ds.add_documents(documents)
    assert len(ds.docs) == 2
    assert all(isinstance(doc, BaseDocument) for doc in ds.docs.values())

    # test updating documents
    ds.add_documents(documents)
    assert len(ds.docs) == 2

    # test getting documents
    doc0 = ds.get_document(documents[0].get_doc_id())
    assert doc0 is not None
    assert documents[0].text == doc0.text

    # test deleting documents
    ds.delete_document(documents[0].get_doc_id())
    assert len(ds.docs) == 1


def test_mongo_docstore_save_load(documents: List[Document]) -> None:
    mongo_client = MockMongoClient()
    ds = MongoDocumentStore(mongo_client=mongo_client)  # type: ignore
    ds.add_documents(documents)
    assert len(ds.docs) == 2

    save_dict = ds.to_dict()
    save_dict["mongo_client"] = mongo_client
    ds_loaded = MongoDocumentStore.from_dict(save_dict)
    assert len(ds_loaded.docs) == 2
    assert ds_loaded._collection_name == ds._collection_name
    assert ds_loaded._db_name == ds._db_name


def test_mongo_docstore_save_load_uri(documents: List[Document]) -> None:
    _mock_client = MockMongoClient()
    with patch(
        "pymongo.MongoClient",
        return_value=_mock_client,
    ):
        ds = MongoDocumentStore.from_uri(uri="test_uri")
        ds.add_documents(documents)
        assert len(ds.docs) == 2

        save_dict = ds.to_dict()
        ds_loaded = MongoDocumentStore.from_dict(save_dict)
        assert len(ds_loaded.docs) == 2
        assert ds_loaded._collection_name == ds._collection_name
        assert ds_loaded._db_name == ds._db_name


def test_mongo_docstore_save_load_host_port(documents: List[Document]) -> None:
    _mock_client = MockMongoClient()
    with patch(
        "pymongo.MongoClient",
        return_value=_mock_client,
    ):
        ds = MongoDocumentStore.from_host_and_port(host="test_host", port=8000)
        ds.add_documents(documents)
        assert len(ds.docs) == 2

        save_dict = ds.to_dict()
        ds_loaded = MongoDocumentStore.from_dict(save_dict)
        assert len(ds_loaded.docs) == 2
        assert ds_loaded._collection_name == ds._collection_name
        assert ds_loaded._db_name == ds._db_name


def test_mongo_docstore_hash(documents: List[Document]) -> None:
    mongo_client = MockMongoClient()
    ds = MongoDocumentStore(mongo_client=mongo_client)  # type: ignore

    # Test setting hash
    ds.set_document_hash("test_doc_id", "test_doc_hash")
    doc_hash = ds.get_document_hash("test_doc_id")
    assert doc_hash == "test_doc_hash"

    # Test updating hash
    ds.set_document_hash("test_doc_id", "test_doc_hash_new")
    doc_hash = ds.get_document_hash("test_doc_id")
    assert doc_hash == "test_doc_hash_new"

    # Test getting non-existent
    doc_hash = ds.get_document_hash("test_not_exist")
    assert doc_hash is None
