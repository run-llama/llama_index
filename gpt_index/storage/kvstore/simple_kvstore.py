import json
import os
from typing import Dict, Optional
import logging

from gpt_index.storage.kvstore.types import (
    DEFAULT_COLLECTION,
    BaseInMemoryKVStore,
)


logger = logging.getLogger(__name__)


class SimpleKVStore(BaseInMemoryKVStore):
    def __init__(self, persist_path: str) -> None:
        self._data: Dict[str, Dict[str, dict]] = {}
        self._persist_path = persist_path

        self.load()

    @property
    def persist_path(self) -> str:
        return self._persist_path

    def put(self, key: str, val: dict, collection: str = DEFAULT_COLLECTION) -> None:
        if collection not in self._data:
            self._data[collection] = {}
        self._data[collection][key] = val

    def get(self, key: str, collection: str = DEFAULT_COLLECTION) -> Optional[dict]:
        collection = self._data.get(collection, None)
        if not collection:
            return None
        return collection.get(key, None)

    def get_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        return self._data.get(collection, {})

    def delete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        try:
            self._data[collection].pop(key)
            return True
        except KeyError:
            return False

    def persist(self) -> None:
        dirpath = os.path.dirname(self._persist_path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        with open(self._persist_path, "w+") as f:
            json.dump(self._data, f)

    def load(self) -> None:
        if os.path.exists(self._persist_path):
            logger.info(f"Loading {__name__} from {self._persist_path}.")
            with open(self._persist_path, "r+") as f:
                self._data = json.load(f)
        else:
            logger.info(
                f"No existing {__name__} found at {self._persist_path}, skipping load."
            )
