"""Base object types."""

import pickle
import warnings
from typing import Any, Callable, Dict, Generic, List, Optional, Sequence, Type, TypeVar

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.query_pipeline.query import (
    ChainableMixin,
    InputKeys,
    OutputKeys,
    QueryComponent,
    validate_and_convert_stringable,
)
from llama_index.core.bridge.pydantic import Field, ConfigDict
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.indices.base import BaseIndex
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.objects.base_node_mapping import (
    DEFAULT_PERSIST_FNAME,
    BaseObjectNodeMapping,
    SimpleObjectNodeMapping,
)
from llama_index.core.schema import BaseNode, QueryBundle, QueryType
from llama_index.core.storage.storage_context import (
    DEFAULT_PERSIST_DIR,
    StorageContext,
)

OT = TypeVar("OT")


class ObjectRetriever(ChainableMixin, Generic[OT]):
    """Object retriever."""

    def __init__(
        self,
        retriever: BaseRetriever,
        object_node_mapping: BaseObjectNodeMapping[OT],
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
    ):
        self._retriever = retriever
        self._object_node_mapping = object_node_mapping
        self._node_postprocessors = node_postprocessors or []

    @property
    def retriever(self) -> BaseRetriever:
        """Retriever."""
        return self._retriever

    @property
    def object_node_mapping(self) -> BaseObjectNodeMapping[OT]:
        """Object node mapping."""
        return self._object_node_mapping

    @property
    def node_postprocessors(self) -> List[BaseNodePostprocessor]:
        """Node postprocessors."""
        return self._node_postprocessors

    def retrieve(self, str_or_query_bundle: QueryType) -> List[OT]:
        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(query_str=str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle

        nodes = self._retriever.retrieve(query_bundle)
        for node_postprocessor in self._node_postprocessors:
            nodes = node_postprocessor.postprocess_nodes(
                nodes, query_bundle=query_bundle
            )

        return [self._object_node_mapping.from_node(node.node) for node in nodes]

    async def aretrieve(self, str_or_query_bundle: QueryType) -> List[OT]:
        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(query_str=str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle

        nodes = await self._retriever.aretrieve(query_bundle)
        for node_postprocessor in self._node_postprocessors:
            nodes = node_postprocessor.postprocess_nodes(
                nodes, query_bundle=query_bundle
            )

        return [self._object_node_mapping.from_node(node.node) for node in nodes]

    def _as_query_component(self, **kwargs: Any) -> QueryComponent:
        """As query component."""
        return ObjectRetrieverComponent(retriever=self)


class ObjectRetrieverComponent(QueryComponent):
    """Object retriever component."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    retriever: ObjectRetriever = Field(..., description="Retriever.")

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

    @property
    def index(self) -> BaseIndex:
        """Index."""
        return self._index

    @property
    def object_node_mapping(self) -> BaseObjectNodeMapping:
        """Object node mapping."""
        return self._object_node_mapping

    @classmethod
    def from_objects(
        cls,
        objects: Sequence[OT],
        object_mapping: Optional[BaseObjectNodeMapping] = None,
        from_node_fn: Optional[Callable[[BaseNode], OT]] = None,
        to_node_fn: Optional[Callable[[OT], BaseNode]] = None,
        index_cls: Type[BaseIndex] = VectorStoreIndex,
        **index_kwargs: Any,
    ) -> "ObjectIndex":
        from llama_index.core.objects.utils import get_object_mapping

        # pick the best mapping if not provided
        if object_mapping is None:
            object_mapping = get_object_mapping(
                objects,
                from_node_fn=from_node_fn,
                to_node_fn=to_node_fn,
            )

        nodes = object_mapping.to_nodes(objects)
        index = index_cls(nodes, **index_kwargs)
        return cls(index, object_mapping)

    @classmethod
    def from_objects_and_index(
        cls,
        objects: Sequence[OT],
        index: BaseIndex,
        object_mapping: Optional[BaseObjectNodeMapping] = None,
        from_node_fn: Optional[Callable[[BaseNode], OT]] = None,
        to_node_fn: Optional[Callable[[OT], BaseNode]] = None,
    ) -> "ObjectIndex":
        from llama_index.core.objects.utils import get_object_mapping

        # pick the best mapping if not provided
        if object_mapping is None:
            object_mapping = get_object_mapping(
                objects,
                from_node_fn=from_node_fn,
                to_node_fn=to_node_fn,
            )

        return cls(index, object_mapping)

    def insert_object(self, obj: Any) -> None:
        self._object_node_mapping.add_object(obj)
        node = self._object_node_mapping.to_node(obj)
        self._index.insert_nodes([node])

    def as_retriever(
        self,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        **kwargs: Any,
    ) -> ObjectRetriever:
        return ObjectRetriever(
            retriever=self._index.as_retriever(**kwargs),
            object_node_mapping=self._object_node_mapping,
            node_postprocessors=node_postprocessors,
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
        from llama_index.core.indices import load_index_from_storage

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
