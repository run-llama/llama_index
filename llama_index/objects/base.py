"""Base object types."""

import pickle
import warnings
from typing import Any, Dict, Generic, List, Optional, Sequence, Type, TypeVar

from llama_index.bridge.pydantic import Field
from llama_index.callbacks.base import CallbackManager
from llama_index.core.base_retriever import BaseRetriever
from llama_index.core.query_pipeline.query_component import (
    ChainableMixin,
    InputKeys,
    OutputKeys,
    QueryComponent,
    validate_and_convert_stringable,
)
from llama_index.indices.base import BaseIndex
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.objects.base_node_mapping import (
    DEFAULT_PERSIST_FNAME,
    BaseObjectNodeMapping,
    SimpleObjectNodeMapping,
)
from llama_index.schema import QueryType
from llama_index.storage.storage_context import DEFAULT_PERSIST_DIR, StorageContext

OT = TypeVar("OT")


class ObjectRetriever(ChainableMixin, Generic[OT]):
    """Object retriever."""

    def __init__(
        self, retriever: BaseRetriever, object_node_mapping: BaseObjectNodeMapping[OT]
    ):
        self._retriever = retriever
        self._object_node_mapping = object_node_mapping

    @property
    def retriever(self) -> BaseRetriever:
        """Retriever."""
        return self._retriever

    def retrieve(self, str_or_query_bundle: QueryType) -> List[OT]:
        nodes = self._retriever.retrieve(str_or_query_bundle)
        return [self._object_node_mapping.from_node(node.node) for node in nodes]

    async def aretrieve(self, str_or_query_bundle: QueryType) -> List[OT]:
        nodes = await self._retriever.aretrieve(str_or_query_bundle)
        return [self._object_node_mapping.from_node(node.node) for node in nodes]

    def _as_query_component(self, **kwargs: Any) -> QueryComponent:
        """As query component."""
        return ObjectRetrieverComponent(retriever=self)


class ObjectRetrieverComponent(QueryComponent):
    """Object retriever component."""

    retriever: ObjectRetriever = Field(..., description="Retriever.")

    class Config:
        arbitrary_types_allowed = True

    def set_callback_manager(self, callback_manager: CallbackManager) -> None:
        """Set callback manager."""
        self.retriever.retriever.callback_manager = callback_manager

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        # make sure input is a string
        input["input"] = validate_and_convert_stringable(input["input"])
        return input

    def _run_component(self, **kwargs: Any) -> Any:
        """Run component."""
        output = self.retriever.retrieve(kwargs["input"])
        return {"output": output}

    async def _arun_component(self, **kwargs: Any) -> Any:
        """Run component (async)."""
        output = await self.retriever.aretrieve(kwargs["input"])
        return {"output": output}

    @property
    def input_keys(self) -> InputKeys:
        """Input keys."""
        return InputKeys.from_keys({"input"})

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        return OutputKeys.from_keys({"output"})


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
        obj_node_mapping_fname: str = DEFAULT_PERSIST_FNAME,
    ) -> None:
        # try to persist object node mapping
        try:
            self._object_node_mapping.persist(
                persist_dir=persist_dir, obj_node_mapping_fname=obj_node_mapping_fname
            )
        except (NotImplementedError, pickle.PickleError) as err:
            warnings.warn(
                (
                    "Unable to persist ObjectNodeMapping. You will need to "
                    "reconstruct the same object node mapping to build this ObjectIndex"
                ),
                stacklevel=2,
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
            # assume SimpleObjectNodeMapping for simplicity as its only subclass
            # that supports this method
            try:
                object_node_mapping = SimpleObjectNodeMapping.from_persist_dir(
                    persist_dir=persist_dir
                )
            except Exception as err:
                raise Exception(
                    "Unable to load from persist dir. The object_node_mapping cannot be loaded."
                ) from err
            else:
                return cls(index=index, object_node_mapping=object_node_mapping)
