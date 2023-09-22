import logging
import os
import pickle
from typing import Dict, Optional

import fsspec

from llama_index.storage.kvstore.types import DEFAULT_COLLECTION, BaseInMemoryKVStore

logger = logging.getLogger(__name__)

DATA_TYPE = Dict[str, Dict[str, dict]]


class SimplePickledKVStore(BaseInMemoryKVStore):
    """Simple in-memory Key-Value store that use pickle as a storing mechanism.

    Args:
        data (Optional[DATA_TYPE]): data to initialize the store with
    """

    def __init__(
        self,
        data: Optional[DATA_TYPE] = None,
    ) -> None:
        """Init a SimplePickledKVStore."""
        self._data: DATA_TYPE = data or {}

    def put(self, key: str, val: dict, collection: str = DEFAULT_COLLECTION) -> None:
        """Put a key-value pair into the store."""
        if collection not in self._data:
            self._data[collection] = {}
        self._data[collection][key] = val.copy()

    def get(self, key: str, collection: str = DEFAULT_COLLECTION) -> Optional[dict]:
        """Get a value from the store."""
        collection_data = self._data.get(collection, None)
        if not collection_data:
            return None
        if key not in collection_data:
            return None
        return collection_data[key].copy()

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

    def persist(
        self, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None
    ) -> None:
        """Persist the store."""
        fs = fs or fsspec.filesystem("file")
        dirpath = os.path.dirname(persist_path)
        if not fs.exists(dirpath):
            fs.makedirs(dirpath)

        with fs.open(persist_path, "wb") as f:
            pickle.dump(self._data, f)

    @classmethod
    def from_persist_path(
        cls, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None
    ) -> "SimplePickledKVStore":
        """Load a SimplePickledKVStore from a persist path and filesystem."""
        fs = fs or fsspec.filesystem("file")
        logger.debug(f"Loading {__name__} from {persist_path}.")
        with fs.open(persist_path, "rb") as f:
            data = pickle.load(f)
        return cls(data)

    def to_dict(self) -> dict:
        """Save the store as dict."""
        return self._data

    @classmethod
    def from_dict(cls, save_dict: dict) -> "SimplePickledKVStore":
        """Load a SimplePickledKVStore from dict."""
        return cls(save_dict)
