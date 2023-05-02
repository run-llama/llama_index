from collections import defaultdict
from typing import Any, Dict, List, Optional
from unittest.mock import Mock
import uuid


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
        _id = _id or obj.get("_id", None) or str(uuid.uuid4())
        obj = obj.copy()
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
        self._collections: Dict[str, MockMongoCollection] = defaultdict(
            MockMongoCollection
        )

    def __getitem__(self, collection: str) -> MockMongoCollection:
        return self._collections[collection]


class MockMongoClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._db = MockMongoDB()

    def __getitem__(self, db: str) -> MockMongoDB:
        del db
        return self._db
