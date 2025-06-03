from abc import ABC, abstractmethod
from collections.abc import MutableMapping, Callable
from typing import Dict, List, Optional, Tuple, TypeVar, Generic

import fsspec

DEFAULT_COLLECTION = "data"
DEFAULT_BATCH_SIZE = 1


class BaseKVStore(ABC):
    """Base key-value store."""

    @abstractmethod
    def put(self, key: str, val: dict, collection: str = DEFAULT_COLLECTION) -> None:
        pass

    @abstractmethod
    async def aput(
        self, key: str, val: dict, collection: str = DEFAULT_COLLECTION
    ) -> None:
        pass

    def put_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        # by default, support a batch size of 1
        if batch_size != 1:
            raise NotImplementedError("Batching not supported by this key-value store.")
        else:
            for key, val in kv_pairs:
                self.put(key, val, collection=collection)

    async def aput_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        # by default, support a batch size of 1
        if batch_size != 1:
            raise NotImplementedError("Batching not supported by this key-value store.")
        else:
            for key, val in kv_pairs:
                await self.aput(key, val, collection=collection)

    @abstractmethod
    def get(self, key: str, collection: str = DEFAULT_COLLECTION) -> Optional[dict]:
        pass

    @abstractmethod
    async def aget(
        self, key: str, collection: str = DEFAULT_COLLECTION
    ) -> Optional[dict]:
        pass

    @abstractmethod
    def get_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        pass

    @abstractmethod
    async def aget_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        pass

    @abstractmethod
    def delete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        pass

    @abstractmethod
    async def adelete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        pass


class BaseInMemoryKVStore(BaseKVStore):
    """Base in-memory key-value store."""

    @abstractmethod
    def persist(
        self, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None
    ) -> None:
        pass

    @classmethod
    @abstractmethod
    def from_persist_path(cls, persist_path: str) -> "BaseInMemoryKVStore":
        """Create a BaseInMemoryKVStore from a persist directory."""


MutableMappingT = TypeVar("MutableMappingT", bound=MutableMapping[str, dict])


class MutableMappingKVStore(Generic[MutableMappingT], BaseKVStore):
    """
    MutableMapping Key-Value store.

    Args:
        mapping_factory (Callable[[], MutableMapping[str, dict]): the mutable mapping factory

    """

    def __init__(self, mapping_factory: Callable[[], MutableMappingT]) -> None:
        """Initialize a MutableMappingKVStore."""
        self._collections_mappings: Dict[str, MutableMappingT] = {}
        self._mapping_factory = mapping_factory

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["factory_fn"] = {"fn": self._mapping_factory}
        del state["_mapping_factory"]
        return state

    def __setstate__(self, state: dict) -> None:
        self._collections_mappings = state["_collections_mappings"]
        self._mapping_factory = state["factory_fn"]["fn"]

    def _get_collection_mapping(self, collection: str) -> MutableMappingT:
        """Get a collection mapping. Create one if it does not exist."""
        if collection not in self._collections_mappings:
            self._collections_mappings[collection] = self._mapping_factory()
        return self._collections_mappings[collection]

    def put(self, key: str, val: dict, collection: str = DEFAULT_COLLECTION) -> None:
        """Put a key-value pair into the store."""
        self._get_collection_mapping(collection)[key] = val.copy()

    async def aput(
        self, key: str, val: dict, collection: str = DEFAULT_COLLECTION
    ) -> None:
        """Put a key-value pair into the store."""
        self.put(key, val, collection=collection)

    def get(self, key: str, collection: str = DEFAULT_COLLECTION) -> Optional[dict]:
        """Get a value from the store."""
        mapping = self._get_collection_mapping(collection)

        if key not in mapping:
            return None
        return mapping[key].copy()

    async def aget(
        self, key: str, collection: str = DEFAULT_COLLECTION
    ) -> Optional[dict]:
        """Get a value from the store."""
        return self.get(key, collection=collection)

    def get_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        """Get all values from the store."""
        return dict(self._get_collection_mapping(collection))

    async def aget_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        """Get all values from the store."""
        return self.get_all(collection=collection)

    def delete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """Delete a value from the store."""
        try:
            self._get_collection_mapping(collection).pop(key)
            return True
        except KeyError:
            return False

    async def adelete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """Delete a value from the store."""
        return self.delete(key, collection=collection)

    # this method is here to avoid TypeChecker shows an error
    def persist(
        self, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None
    ) -> None:
        """Persist the store."""
        raise NotImplementedError(
            "Use subclasses of MutableMappingKVStore (such as SimpleKVStore) to call this method"
        )

    # this method is here to avoid TypeChecker shows an error
    def from_persist_path(cls, persist_path: str) -> "MutableMappingKVStore":
        """Create a MutableMappingKVStore from a persist directory."""
        raise NotImplementedError(
            "Use subclasses of MutableMappingKVStore (such as SimpleKVStore) to call this method"
        )
