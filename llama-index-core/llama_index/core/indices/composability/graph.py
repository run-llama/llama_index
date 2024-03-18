"""Composability graphs."""

from typing import Any, Dict, List, Optional, Sequence, Type, cast

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.data_structs.data_structs import IndexStruct
from llama_index.core.indices.base import BaseIndex
from llama_index.core.schema import (
    IndexNode,
    NodeRelationship,
    ObjectType,
    RelatedNodeInfo,
)
from llama_index.core.service_context import ServiceContext
from llama_index.core.storage.storage_context import StorageContext


class ComposableGraph:
    """Composable graph."""

    def __init__(
        self,
        all_indices: Dict[str, BaseIndex],
        root_id: str,
        storage_context: Optional[StorageContext] = None,
    ) -> None:
        """Init params."""
        self._all_indices = all_indices
        self._root_id = root_id
        self.storage_context = storage_context

    @property
    def root_id(self) -> str:
        return self._root_id

    @property
    def all_indices(self) -> Dict[str, BaseIndex]:
        return self._all_indices

    @property
    def root_index(self) -> BaseIndex:
        return self._all_indices[self._root_id]

    @property
    def index_struct(self) -> IndexStruct:
        return self._all_indices[self._root_id].index_struct

    @property
    def service_context(self) -> Optional[ServiceContext]:
        return self._all_indices[self._root_id].service_context

    @classmethod
    def from_indices(
        cls,
        root_index_cls: Type[BaseIndex],
        children_indices: Sequence[BaseIndex],
        index_summaries: Optional[Sequence[str]] = None,
        service_context: Optional[ServiceContext] = None,
        storage_context: Optional[StorageContext] = None,
        **kwargs: Any,
    ) -> "ComposableGraph":  # type: ignore
        """Create composable graph using this index class as the root."""
        service_context = service_context or ServiceContext.from_defaults()
        with service_context.callback_manager.as_trace("graph_construction"):
            if index_summaries is None:
                for index in children_indices:
                    if index.index_struct.summary is None:
                        raise ValueError(
                            "Summary must be set for children indices. "
                            "If the index does a summary "
                            "(through index.index_struct.summary), then "
                            "it must be specified with then `index_summaries` "
                            "argument in this function. We will support "
                            "automatically setting the summary in the future."
                        )
                index_summaries = [
                    index.index_struct.summary for index in children_indices
                ]
            else:
                # set summaries for each index
                for index, summary in zip(children_indices, index_summaries):
                    index.index_struct.summary = summary

            if len(children_indices) != len(index_summaries):
                raise ValueError("indices and index_summaries must have same length!")

            # construct index nodes
            index_nodes = []
            for index, summary in zip(children_indices, index_summaries):
                assert isinstance(index.index_struct, IndexStruct)
                index_node = IndexNode(
                    text=summary,
                    index_id=index.index_id,
                    relationships={
                        NodeRelationship.SOURCE: RelatedNodeInfo(
                            node_id=index.index_id, node_type=ObjectType.INDEX
                        )
                    },
                )
                index_nodes.append(index_node)

            # construct root index
            root_index = root_index_cls(
                nodes=index_nodes,
                service_context=service_context,
                storage_context=storage_context,
                **kwargs,
            )
            # type: ignore
            all_indices: List[BaseIndex] = [
                *cast(List[BaseIndex], children_indices),
                root_index,
            ]

            return cls(
                all_indices={index.index_id: index for index in all_indices},
                root_id=root_index.index_id,
                storage_context=storage_context,
            )

    def get_index(self, index_struct_id: Optional[str] = None) -> BaseIndex:
        """Get index from index struct id."""
        if index_struct_id is None:
            index_struct_id = self._root_id
        return self._all_indices[index_struct_id]

    def as_query_engine(self, **kwargs: Any) -> BaseQueryEngine:
        # NOTE: lazy import
        from llama_index.core.query_engine.graph_query_engine import (
            ComposableGraphQueryEngine,
        )

        return ComposableGraphQueryEngine(self, **kwargs)
