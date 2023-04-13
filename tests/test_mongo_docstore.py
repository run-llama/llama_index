from typing import Any, List, Optional
from unittest.mock import Mock
import uuid

import pytest

from gpt_index.docstore.mongo_docstore import MongoDocumentStore
from gpt_index.readers.schema.base import Document


class MockMongoCollection:
    def __init__(self) -> None:
        self._data = {}

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

    def insert_many(self, objs: List[dict]) -> Any:
        results = [self.insert_one(obj) for obj in objs]
        inserted_ids = [result.inserted_id for result in results]

        insert_result = Mock()
        insert_result.inserted_ids = inserted_ids
        return insert_result


class MockMongoDB:
    def __init__(self) -> None:
        self._collection = MockMongoCollection()

    def __getitem__(self, collection: str) -> MockMongoCollection:
        del collection
        return self._collection


class MockMongoClient:
    def __init__(self) -> None:
        self._db = MockMongoDB()
        pass

    def __getitem__(self, db: str) -> MockMongoDB:
        del db
        return self._db


@pytest.fixture
def documents() -> List[Document]:
    return [
        Document("doc_1"),
        Document("doc_2"),
    ]


def test_mongo_docstore(documents: List[Document]):
    ds = MongoDocumentStore(client=MockMongoClient())
    assert len(ds.docs) == 0

    # test adding documents
    ds.add_documents(documents)
    assert len(ds.docs) == 2

    # test updating documents
    ds.add_documents(documents)
    assert len(ds.docs) == 2

    # test getting documents
    doc0 = ds.get_document(documents[0].doc_id)
    assert documents[0].text == doc0.text

    # test deleting documents
    ds.delete_document(documents[0].doc_id)
    assert len(ds.docs) == 1
