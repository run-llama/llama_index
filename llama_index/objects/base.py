"""Base object types."""

from typing import Any, Generic, List, Optional, Sequence, Type, TypeVar

from llama_index.core import BaseRetriever
from llama_index.indices.base import BaseIndex
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.objects.base_node_mapping import (
    BaseObjectNodeMapping,
    SimpleObjectNodeMapping,
)
from llama_index.schema import QueryType

OT = TypeVar("OT")


class ObjectRetriever(Generic[OT]):
    """Object retriever."""

    def __init__(
        self, retriever: BaseRetriever, object_node_mapping: BaseObjectNodeMapping[OT]
    ):
        self._retriever = retriever
        self._object_node_mapping = object_node_mapping

    def retrieve(self, str_or_query_bundle: QueryType) -> List[OT]:
        nodes = self._retriever.retrieve(str_or_query_bundle)
        return [self._object_node_mapping.from_node(node.node) for node in nodes]


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
        object_mapping: Optional[BaseObjectNodeMapping] = None,
        index_cls: Type[BaseIndex] = VectorStoreIndex,
        **index_kwargs: Any,
    ) -> "ObjectIndex":
        if object_mapping is None:
            object_mapping = SimpleObjectNodeMapping.from_objects(objects)
        nodes = object_mapping.to_nodes(objects)
        index = index_cls(nodes, **index_kwargs)
        return cls(index, object_mapping)

    def insert_object(self, obj: Any) -> None:
        self._object_node_mapping.add_object(obj)
        node = self._object_node_mapping.to_node(obj)
        self._index.insert_nodes([node])

    def as_retriever(self, **kwargs: Any) -> ObjectRetriever:
        return ObjectRetriever(
            retriever=self._index.as_retriever(**kwargs),
            object_node_mapping=self._object_node_mapping,
        )

    def as_node_retriever(self, **kwargs: Any) -> BaseRetriever:
        return self._index.as_retriever(**kwargs)
