import json
import logging
import os
from typing import Dict, Optional

import fsspec
from llama_index.core.storage.kvstore.types import (
    MutableMappingKVStore,
)

logger = logging.getLogger(__name__)

DATA_TYPE = Dict[str, Dict[str, dict]]


class SimpleKVStore(MutableMappingKVStore[dict]):
    """
    Simple in-memory Key-Value store.

    Args:
        data (Optional[DATA_TYPE]): data to initialize the store with

    """

    def __init__(
        self,
        data: Optional[DATA_TYPE] = None,
    ) -> None:
        """Init a SimpleKVStore."""
        super().__init__(mapping_factory=dict)

        if data is not None:
            self._collections_mappings = data.copy()

    def persist(
        self, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None
    ) -> None:
        """Persist the store."""
        fs = fs or fsspec.filesystem("file")
        dirpath = os.path.dirname(persist_path)
        if not fs.exists(dirpath):
            fs.makedirs(dirpath)

        with fs.open(persist_path, "w") as f:
            f.write(json.dumps(self._collections_mappings))

    @classmethod
    def from_persist_path(
        cls, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None
    ) -> "SimpleKVStore":
        """Load a SimpleKVStore from a persist path and filesystem."""
        fs = fs or fsspec.filesystem("file")
        logger.debug(f"Loading {__name__} from {persist_path}.")
        with fs.open(persist_path, "rb") as f:
            data = json.load(f)
        return cls(data)

    def to_dict(self) -> dict:
        """Save the store as dict."""
        return self._collections_mappings.copy()

    @classmethod
    def from_dict(cls, save_dict: dict) -> "SimpleKVStore":
        """Load a SimpleKVStore from dict."""
        return cls(save_dict)
