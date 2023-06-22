"""Base object types."""

from typing import TypeVar, Generic, Sequence, Type, Any, List
from abc import abstractmethod
from llama_index.data_structs.node import Node
from llama_index.indices.base import BaseIndex
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.indices.objects.node_mapping import (
    BaseObjectNodeMapping,
    SimpleObjectNodeMapping,
    OT,
)


class ObjectRetriever(Generic[OT]):
    def __init__(
        self, retriever: BaseRetriever, object_node_mapping: BaseObjectNodeMapping[OT]
    ):
        self._retriever = retriever
        self._object_node_mapping = object_node_mapping

    def retrieve(self, query: str) -> List[OT]:
        nodes = self._retriever.retrieve(query)
        return [self._object_node_mapping.from_node(node) for node in nodes]


class ObjectIndex(Generic[OT]):
    """Object index."""

    def __init__(
        self, index: BaseIndex, object_node_mapping: BaseObjectNodeMapping
    ) -> None:
        self._index = index
        self._object_node_mapping = object_node_mapping

    @classmethod
    def from_objects(
        cls,
        objects: Sequence[OT],
        index_cls: Type[BaseIndex] = VectorStoreIndex,
        object_mapping_cls: Type[BaseObjectNodeMapping] = SimpleObjectNodeMapping,
        **index_kwargs: Any,
    ) -> "ObjectIndex":
        object_node_mapping = object_mapping_cls(objects)
        nodes = object_node_mapping.to_nodes()
        return index_cls(nodes, **index_kwargs)

    def insert_object(self, obj: Any):
        self._object_node_mapping.add_object(obj)
        node = self._object_node_mapping.to_node(obj)
        self._index.insert_nodes([node])

    def as_retriever(self, **kwargs) -> ObjectRetriever:
        return ObjectRetriever(
            retriever=self.as_retriever(**kwargs),
            object_node_mapping=self._object_node_mapping,
        )
