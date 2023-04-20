from typing import Dict, Optional

from gpt_index.storage.keyval_store.types import BaseKeyValStore

DEFAULT_COLLECTION = "data"


class InMemoryKeyValStore(BaseKeyValStore):
    def __init__(self) -> None:
        self.data: Dict[str, Dict[str, dict]] = {}

    def add(self, key: str, val: dict, collection: str = DEFAULT_COLLECTION) -> None:
        if collection not in self.data:
            self.data[collection] = {}
        self.data[key] = val

    def get(self, key: str, collection: str = DEFAULT_COLLECTION) -> Optional[dict]:
        collection = self.data.get(collection, None)
        if not collection:
            return None
        return collection.get(key, None)

    def get_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        return self.data.get(collection, {})

    def delete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        try:
            self.data[collection].pop(key)
            return True
        except KeyError:
            return False

    def save_to_disk(self, path: str) -> None:
        pass

    @classmethod
    def load_from_disk(self, path: str) -> None:
        pass
