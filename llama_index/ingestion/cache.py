from typing import List, Optional

import fsspec

from llama_index.bridge.pydantic import BaseModel, Field
from llama_index.schema import BaseNode, Document, ImageNode, IndexNode, TextNode
from llama_index.storage.kvstore import (
    FirestoreKVStore as FirestoreCache,
)
from llama_index.storage.kvstore import (
    MongoDBKVStore as MongoDBCache,
)
from llama_index.storage.kvstore import (
    RedisKVStore as RedisCache,
)
from llama_index.storage.kvstore import (
    SimpleKVStore as SimpleCache,
)
from llama_index.storage.kvstore.types import (
    BaseKVStore as BaseCache,
)

DEFAULT_CACHE_NAME = "llama_cache"


class IngestionCache(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    nodes_key = "nodes"

    collection: str = Field(
        default=DEFAULT_CACHE_NAME, description="Collection name of the cache."
    )
    cache: BaseCache = Field(default_factory=SimpleCache, description="Cache to use.")

    def _load_nodes(self, node_dicts: List[dict]) -> List[BaseNode]:
        nodes: List[BaseNode] = []

        for node_dict in node_dicts:
            class_name = node_dict.get("class_name", None)
            if class_name == TextNode.class_name():
                nodes.append(TextNode.from_dict({**node_dict}))
            elif class_name == ImageNode.class_name():
                nodes.append(ImageNode.from_dict({**node_dict}))
            elif class_name == IndexNode.class_name():
                nodes.append(IndexNode.from_dict({**node_dict}))
            elif class_name == Document.class_name():
                nodes.append(Document.from_dict({**node_dict}))
            else:
                raise ValueError(f"Unknown node class name: {class_name}")

        return nodes

    # TODO: add async get/put methods?
    def put(
        self, key: str, nodes: List[BaseNode], collection: Optional[str] = None
    ) -> None:
        """Put a value into the cache."""
        collection = collection or self.collection

        val = {self.nodes_key: [node.to_dict() for node in nodes]}
        self.cache.put(key, val, collection=collection)

    def get(
        self, key: str, collection: Optional[str] = None
    ) -> Optional[List[BaseNode]]:
        """Get a value from the cache."""
        collection = collection or self.collection
        node_dicts = self.cache.get(key, collection=collection)

        if node_dicts is None:
            return None

        return self._load_nodes(node_dicts[self.nodes_key])

    def persist(
        self, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None
    ) -> None:
        """Persist the cache to a directory, if possible."""
        if isinstance(self.cache, SimpleCache):
            self.cache.persist(persist_path, fs=fs)
        else:
            print("Warning: skipping persist, only needed for SimpleCache.")

    @classmethod
    def from_persist_path(
        cls, persist_path: str, collection: str = DEFAULT_CACHE_NAME
    ) -> "IngestionCache":
        """Create a IngestionCache from a persist directory."""
        return cls(
            collection=collection,
            cache=SimpleCache.from_persist_path(persist_path),
        )


__all__ = [
    "SimpleCache",
    "RedisCache",
    "MongoDBCache",
    "FirestoreCache",
]
