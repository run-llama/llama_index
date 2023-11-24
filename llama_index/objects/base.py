"""Base object types."""

import pickle
import warnings
from typing import Any, Generic, List, Optional, Sequence, Type, TypeVar

from llama_index.core import BaseRetriever
from llama_index.indices.base import BaseIndex
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.objects.base_node_mapping import (
    BaseObjectNodeMapping,
    SimpleObjectNodeMapping,
)
from llama_index.schema import QueryType
from llama_index.storage.storage_context import DEFAULT_PERSIST_DIR, StorageContext

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

    def persist(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
    ) -> None:
        # try to persist object node mapping
        try:
            self._object_node_mapping.persist(persist_dir=persist_dir)
        except (NotImplementedError, pickle.PickleError) as err:
            warnings.warn(
                (
                    "Unable to persist ObjectNodeMapping. You will need to "
                    "reconstruct the same object node mapping to build this ObjectIndex"
                ),
                DeprecationWarning,
            )
        self._index._storage_context.persist(persist_dir=persist_dir)

    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        object_node_mapping: Optional[BaseObjectNodeMapping] = None,
    ) -> "ObjectIndex":
        from llama_index.indices import load_index_from_storage

        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        if object_node_mapping:
            return cls(index=index, object_node_mapping=object_node_mapping)
        else:
            # try to load object_node_mapping
            try:
                object_node_mapping = cls._resolve_object_node_mapping()
            except (NotImplementedError, pickle.PickleError) as err:
                raise ValueError(
                    "Unable to load from persist dir. The object_node_mapping cannot be loaded."
                ) from err
            else:
                return cls(index=index, object_node_mapping=object_node_mapping)

    @staticmethod
    def _resolve_object_node_mapping(
        persist_dir: str = DEFAULT_PERSIST_DIR,
    ) -> Type[BaseObjectNodeMapping]:
        from llama_index.objects import (
            SimpleObjectNodeMapping,
            SimpleQueryToolNodeMapping,
            SimpleToolNodeMapping,
            SQLTableNodeMapping,
        )

        object_node_mapping = None
        error = None
        for object_node_mapping_type in [
            SimpleObjectNodeMapping,
            SimpleToolNodeMapping,
            SimpleQueryToolNodeMapping,
            SQLTableNodeMapping,
        ]:
            try:
                object_node_mapping = object_node_mapping_type.from_persist_dir(
                    persist_dir=persist_dir
                )
                break
            except (NotImplementedError, pickle.PickleError) as err:
                error = err

        if object_node_mapping:
            return object_node_mapping
        else:
            raise ValueError("Unable to load object_node_mapping.") from error
