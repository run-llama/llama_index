"""Composability graphs."""

import json
from typing import Any, Dict, List, Optional, Sequence, Union

from gpt_index.constants import DOCSTORE_KEY, INDEX_STRUCT_KEY
from gpt_index.data_structs.data_structs_v2 import CompositeIndex
from gpt_index.data_structs.data_structs_v2 import V2IndexStruct as IndexStruct
from gpt_index.docstore import DocumentStore
from gpt_index.indices.query.query_runner import QueryRunner
from gpt_index.indices.query.query_transform.base import BaseQueryTransform
from gpt_index.indices.query.schema import QueryBundle, QueryConfig
from gpt_index.indices.registry import load_index_struct_from_dict
from gpt_index.indices.service_context import ServiceContext
from gpt_index.response.schema import Response

# TMP: refactor query config type
QUERY_CONFIG_TYPE = Union[Dict, QueryConfig]

class ComposableGraph:
    """Composable graph."""

    def __init__(
        self,
        index_struct: CompositeIndex,
        docstore: DocumentStore,
        service_context: Optional[ServiceContext] = None,
    ) -> None:
        """Init params."""
        self._docstore = docstore
        self._index_struct = index_struct
        self._service_context = service_context or ServiceContext.from_defaults()

    @classmethod
    def from_indices(cls, all_indices: Dict[str, IndexStruct], root_id: str, docstores: Sequence[DocumentStore]):
        composite_index_struct = CompositeIndex(
            all_indices=all_indices,
            root_id=root_id,
        )
        merged_docstore = DocumentStore.merge(docstores)
        return cls(index_struct=composite_index_struct, docstore=merged_docstore)

    def query(
        self,
        query_str: Union[str, QueryBundle],
        query_configs: Optional[List[QUERY_CONFIG_TYPE]] = None,
        query_transform: Optional[BaseQueryTransform] = None,
    ) -> Response:
        """Query the index."""
        query_runner = QueryRunner(
            self._service_context,
            self._docstore,
            query_configs=query_configs,
            query_transform=query_transform,
            recursive=True,
        )
        return query_runner.query(query_str, self._index_struct)

    async def aquery(
        self,
        query_str: Union[str, QueryBundle],
        query_configs: Optional[List[QUERY_CONFIG_TYPE]] = None,
        query_transform: Optional[BaseQueryTransform] = None,
    ) -> Response:
        """Query the index."""
        query_runner = QueryRunner(
            self._service_context,
            self._docstore,
            query_configs=query_configs,
            query_transform=query_transform,
            recursive=True,
        )
        return await query_runner.aquery(query_str, self._index_struct)

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
        result_dict = json.loads(index_string)
        index_struct = load_index_struct_from_dict(result_dict['index_struct'])
        docstore = DocumentStore.load_from_dict(result_dict["docstore"])
        return cls(index_struct, docstore, **kwargs)

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
