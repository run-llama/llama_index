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
    """Simple in-memory Key-Value store.

    Args:
        persist_path (str): path to persist the store

    """

    def __init__(self, persist_path: str) -> None:
        """Init a SimpleKVStore."""
        self._data: Dict[str, Dict[str, dict]] = {}
        self._persist_path = persist_path

        self.load()

    @property
    def persist_path(self) -> str:
        """Return the path to persist the store."""
        return self._persist_path

    def put(self, key: str, val: dict, collection: str = DEFAULT_COLLECTION) -> None:
        """Put a key-value pair into the store."""
        if collection not in self._data:
            self._data[collection] = {}
        self._data[collection][key] = val.copy()

    def get(self, key: str, collection: str = DEFAULT_COLLECTION) -> Optional[dict]:
        """Get a value from the store."""
        collection = self._data.get(collection, None)
        if not collection:
            return None
        if key not in collection:
            return None
        return collection[key].copy()

    def get_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        """Get all values from the store."""
        return self._data.get(collection, {}).copy()

    def delete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """Delete a value from the store."""
        try:
            self._data[collection].pop(key)
            return True
        except KeyError:
            return False

    def persist(self) -> None:
        """Persist the store."""
        dirpath = os.path.dirname(self._persist_path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        with open(self._persist_path, "w+") as f:
            json.dump(self._data, f)

    def load(self) -> None:
        """Load the store."""
        if os.path.exists(self._persist_path):
            logger.info(f"Loading {__name__} from {self._persist_path}.")
            with open(self._persist_path, "r+") as f:
                self._data = json.load(f)
        else:
            logger.info(
                f"No existing {__name__} found at {self._persist_path}, skipping load."
            )
