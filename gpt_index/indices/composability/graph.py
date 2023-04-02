"""Composability graphs."""

import json
from typing import Any, Dict, List, Optional, Sequence, Type, Union, cast

from gpt_index.constants import (
    ADDITIONAL_QUERY_CONTEXT_KEY,
    DOCSTORE_KEY,
    INDEX_STRUCT_KEY,
    VECTOR_STORE_KEY,
)
from gpt_index.data_structs.data_structs_v2 import CompositeIndex
from gpt_index.data_structs.data_structs_v2 import V2IndexStruct
from gpt_index.data_structs.data_structs_v2 import V2IndexStruct as IndexStruct
from gpt_index.data_structs.node_v2 import IndexNode, DocumentRelationship
from gpt_index.docstore_v2 import DocumentStore
from gpt_index.indices.base import BaseGPTIndex
from gpt_index.indices.query.query_runner import QueryRunner
from gpt_index.indices.query.query_transform.base import BaseQueryTransform
from gpt_index.indices.query.schema import QueryBundle, QueryConfig
from gpt_index.indices.service_context import ServiceContext
from gpt_index.response.schema import RESPONSE_TYPE
from gpt_index.vector_stores.registry import (
    load_vector_store_from_dict,
    save_vector_store_to_dict,
)
from gpt_index.vector_stores.types import VectorStore

# TMP: refactor query config type
QUERY_CONFIG_TYPE = Union[Dict, QueryConfig]


def save_additional_query_context_to_dict(
    additional_query_context: Dict[str, Dict[str, Any]]
):
    """Save additional query context to dict.

    NOTE: Right now, only consider vector stores."""
    save_dict = {}
    for index_id, index_context_dict in additional_query_context.items():
        index_save_dict = {}
        for key, val in index_context_dict.items():
            if isinstance(val, VectorStore):
                index_save_dict[key] = save_vector_store_to_dict(val)
        save_dict[index_id] = index_save_dict
    return save_dict


def load_additional_query_context_from_dict(
    save_dict: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Load additional query context from dict.

    NOTE: Right now, only consider vector stores."""
    context_dict = {}
    for index_id, index_save_dict in save_dict.items():
        index_context_dict = {}
        for key, val in index_save_dict.items():
            if key == VECTOR_STORE_KEY:
                index_context_dict[key] = load_vector_store_from_dict(val)
            else:
                index_context_dict[key] = val
        context_dict[index_id] = index_context_dict
    return context_dict


class ComposableGraph:
    """Composable graph."""

    def __init__(
        self,
        index_struct: CompositeIndex,
        docstore: DocumentStore,
        service_context: Optional[ServiceContext] = None,
        additional_query_context: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """Init params."""
        self._docstore = docstore
        self._index_struct = index_struct
        self._service_context = service_context or ServiceContext.from_defaults()
        self._additional_query_context = additional_query_context or {}

    @property
    def index_struct(self) -> CompositeIndex:
        return self._index_struct

    @property
    def service_context(self) -> ServiceContext:
        return self._service_context

    @classmethod
    def from_index_structs_and_docstores(
        cls,
        all_index_structs: Dict[str, IndexStruct],
        root_id: str,
        docstores: Sequence[DocumentStore],
        additional_query_context: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> "ComposableGraph":
        composite_index_struct = CompositeIndex(
            all_index_structs=all_index_structs,
            root_id=root_id,
        )
        merged_docstore = DocumentStore.merge(docstores)
        return cls(
            index_struct=composite_index_struct,
            docstore=merged_docstore,
            additional_query_context=additional_query_context,
            **kwargs,
        )

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

        # collect additional query context
        additional_query_context: Dict[str, Dict[str, Any]] = {}
        additional_query_context[
            root_index.index_struct.index_id
        ] = root_index.additional_query_context
        for index in children_indices:
            additional_query_context[
                index.index_struct.index_id
            ] = index.additional_query_context

        return cls.from_index_structs_and_docstores(
            all_index_structs={
                index.index_struct.index_id: index.index_struct for index in all_indices
            },
            root_id=root_index.index_struct.index_id,
            docstores=[index.docstore for index in all_indices],
            service_context=root_index.service_context,
            additional_query_context=additional_query_context,
        )

    def query(
        self,
        query_str: Union[str, QueryBundle],
        query_configs: Optional[List[QUERY_CONFIG_TYPE]] = None,
        query_transform: Optional[BaseQueryTransform] = None,
        service_context: Optional[ServiceContext] = None,
    ) -> RESPONSE_TYPE:
        """Query the index."""
        service_context = service_context or self._service_context
        query_runner = QueryRunner(
            index_struct=self._index_struct,
            service_context=service_context,
            additional_query_context=self._additional_query_context,
            docstore=self._docstore,
            query_configs=query_configs,
            query_transform=query_transform,
            recursive=True,
        )
        return query_runner.query(query_str)

    async def aquery(
        self,
        query_str: Union[str, QueryBundle],
        query_configs: Optional[List[QUERY_CONFIG_TYPE]] = None,
        query_transform: Optional[BaseQueryTransform] = None,
        service_context: Optional[ServiceContext] = None,
    ) -> RESPONSE_TYPE:
        """Query the index."""
        service_context = service_context or self._service_context
        query_runner = QueryRunner(
            index_struct=self._index_struct,
            service_context=service_context,
            additional_query_context=self._additional_query_context,
            docstore=self._docstore,
            query_configs=query_configs,
            query_transform=query_transform,
            recursive=True,
        )
        return await query_runner.aquery(query_str)

    def get_index(
        self, index_struct_id: str, index_cls: Type[BaseGPTIndex], **kwargs: Any
    ) -> BaseGPTIndex:
        """Get index from index struct id."""
        index_struct = self._index_struct.all_index_structs[index_struct_id]
        return index_cls(
            index_struct=index_struct,
            docstore=self._docstore,
            **kwargs,
        )

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
        from gpt_index.indices.registry import load_index_struct_from_dict

        result_dict = json.loads(index_string)
        index_struct = load_index_struct_from_dict(result_dict[INDEX_STRUCT_KEY])
        docstore = DocumentStore.load_from_dict(result_dict[DOCSTORE_KEY])
        additional_query_context = load_additional_query_context_from_dict(
            result_dict[ADDITIONAL_QUERY_CONTEXT_KEY]
        )
        assert isinstance(index_struct, CompositeIndex)
        return cls(
            index_struct,
            docstore,
            additional_query_context=additional_query_context,
            **kwargs,
        )

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
            INDEX_STRUCT_KEY: self._index_struct.to_dict(),
            DOCSTORE_KEY: self._docstore.serialize_to_dict(),
            ADDITIONAL_QUERY_CONTEXT_KEY: save_additional_query_context_to_dict(
                self._additional_query_context
            ),
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
