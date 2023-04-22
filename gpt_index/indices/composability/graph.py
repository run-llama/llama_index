"""Composability graphs."""

import json
from typing import Any, Dict, List, Optional, Sequence, Type, Union, cast

from gpt_index.constants import (
    ALL_INDICES_KEY,
    ROOT_INDEX_ID_KEY,
)
from gpt_index.data_structs.data_structs_v2 import CompositeIndex, V2IndexStruct
from gpt_index.data_structs.node_v2 import IndexNode, DocumentRelationship
from gpt_index.indices.base import BaseGPTIndex
from gpt_index.indices.query.base import BaseQueryEngine
from gpt_index.indices.query.graph_query_engine import ComposableGraphQueryEngine
from gpt_index.indices.query.schema import QueryConfig
from gpt_index.indices.registry import save_index_to_dict
from gpt_index.indices.service_context import ServiceContext

# TMP: refactor query config type
QUERY_CONFIG_TYPE = Union[Dict, QueryConfig]


class ComposableGraph:
    """Composable graph."""

    def __init__(
        self,
        all_indices: Dict[str, BaseGPTIndex],
        root_id: str,
    ) -> None:
        """Init params."""
        self._all_indices = all_indices
        self._root_id = root_id

    @property
    def root_id(self) -> str:
        return self._root_id

    @property
    def all_indices(self) -> Dict[str, BaseGPTIndex]:
        return self._all_indices

    @property
    def root_index(self) -> BaseGPTIndex:
        return self._all_indices[self._root_id]

    @property
    def index_struct(self) -> CompositeIndex:
        return self._all_indices[self._root_id].index_struct

    @property
    def service_context(self) -> ServiceContext:
        return self._all_indices[self._root_id].service_context

    @classmethod
    def from_indices(
        cls,
        root_index_cls: Type[BaseGPTIndex],
        children_indices: Sequence[BaseGPTIndex],
        index_summaries: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> "ComposableGraph":  # type: ignore
        """Create composable graph using this index class as the root."""
        if index_summaries is None:
            for index in children_indices:
                if index.index_struct.summary is None:
                    raise ValueError(
                        "Summary must be set for children indices. If the index does "
                        "a summary (through index.index_struct.summary), then it must "
                        "be specified with then `index_summaries` "
                        "argument in this function."
                        "We will support automatically setting the summary in the "
                        "future."
                    )
            index_summaries = [index.index_struct.summary for index in children_indices]
        else:
            # set summaries for each index
            for index, summary in zip(children_indices, index_summaries):
                index.index_struct.summary = summary

        if len(children_indices) != len(index_summaries):
            raise ValueError("indices and index_summaries must have same length!")

        # construct index nodes
        index_nodes = []
        for index, summary in zip(children_indices, index_summaries):
            assert isinstance(index.index_struct, V2IndexStruct)
            index_node = IndexNode(
                text=summary,
                index_id=index.index_struct.index_id,
                relationships={
                    DocumentRelationship.SOURCE: index.index_struct.index_id
                },
            )
            index_nodes.append(index_node)

        # construct root index
        root_index = root_index_cls(
            nodes=index_nodes,
            **kwargs,
        )
        # type: ignore
        all_indices: List[BaseGPTIndex] = cast(List[BaseGPTIndex], children_indices) + [
            root_index
        ]

        return cls(
            all_indices={index.index_struct.index_id: index for index in all_indices},
            root_id=root_index.index_struct.index_id,
        )

    def get_index(self, index_struct_id: Optional[str] = None) -> BaseGPTIndex:
        """Get index from index struct id."""
        if index_struct_id is None:
            index_struct_id = self._root_id
        return self._all_indices[index_struct_id]

    def as_query_engine(self, **kwargs: Any) -> BaseQueryEngine:
        return ComposableGraphQueryEngine(self, **kwargs)

    @classmethod
    def load_from_string(cls, index_string: str, **kwargs: Any) -> "ComposableGraph":
        """Load index from string (in JSON-format).

        This method loads the index from a JSON string. The index data
        structure itself is preserved completely. If the index is defined over
        subindices, those subindices will also be preserved (and subindices of
        those subindices, etc.).

        Args:
            save_path (str): The save_path of the file.

        Returns:
            BaseGPTIndex: The loaded index.

        """
        # lazy load registry
        from gpt_index.indices.registry import load_index_from_dict

        result_dict: Dict[str, Any] = json.loads(index_string)
        all_indices_dict: Dict[str, dict] = result_dict[ALL_INDICES_KEY]
        root_id = result_dict[ROOT_INDEX_ID_KEY]
        index_kwargs = kwargs.get("index_kwargs", {})
        all_indices = {
            index_id: load_index_from_dict(index_dict, **index_kwargs.get(index_id, {}))
            for index_id, index_dict in all_indices_dict.items()
        }
        return cls(all_indices, root_id)

    @classmethod
    def load_from_disk(cls, save_path: str, **kwargs: Any) -> "ComposableGraph":
        """Load index from disk.

        This method loads the index from a JSON file stored on disk. The index data
        structure itself is preserved completely. If the index is defined over
        subindices, those subindices will also be preserved (and subindices of
        those subindices, etc.).

        Args:
            save_path (str): The save_path of the file.

        Returns:
            BaseGPTIndex: The loaded index.

        """
        with open(save_path, "r") as f:
            file_contents = f.read()
            return cls.load_from_string(file_contents, **kwargs)

    def save_to_string(self, **save_kwargs: Any) -> str:
        """Save to string.

        This method stores the index into a JSON file stored on disk.

        Args:
            save_path (str): The save_path of the file.

        """
        out_dict: Dict[str, Any] = {
            ALL_INDICES_KEY: {
                index_id: save_index_to_dict(index)
                for index_id, index in self._all_indices.items()
            },
            ROOT_INDEX_ID_KEY: self._root_id,
        }
        return json.dumps(out_dict)

    def save_to_disk(self, save_path: str, **save_kwargs: Any) -> None:
        """Save to file.

        This method stores the index into a JSON file stored on disk.

        Args:
            save_path (str): The save_path of the file.

        """
        index_string = self.save_to_string(**save_kwargs)
        with open(save_path, "w") as f:
            f.write(index_string)
