"""Composability graphs."""

import json
from typing import Any, Dict, List, Optional, Sequence, Type, Union

from gpt_index.data_structs.data_structs_v2 import V2IndexStruct as IndexStruct
from gpt_index.data_structs.struct_type import IndexStructType
from gpt_index.docstore import DocumentStore
from gpt_index.embeddings.base import BaseEmbedding
from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.indices.base import BaseGPTIndex
from gpt_index.indices.empty.base import GPTEmptyIndex
from gpt_index.indices.keyword_table.base import GPTKeywordTableIndex
from gpt_index.indices.knowledge_graph.base import GPTKnowledgeGraphIndex
from gpt_index.indices.list.base import GPTListIndex
from gpt_index.indices.prompt_helper import PromptHelper
from gpt_index.indices.query.query_runner import QueryRunner
from gpt_index.indices.query.query_transform.base import BaseQueryTransform
from gpt_index.indices.query.schema import QueryBundle, QueryConfig
from gpt_index.indices.registry import IndexRegistry
from gpt_index.indices.service_context import ServiceContext
from gpt_index.indices.struct_store.sql import GPTSQLStructStoreIndex
from gpt_index.indices.tree.base import GPTTreeIndex
from gpt_index.indices.vector_store.base import GPTVectorStoreIndex
from gpt_index.indices.vector_store.vector_indices import (
    GPTChromaIndex,
    GPTFaissIndex,
    GPTPineconeIndex,
    GPTQdrantIndex,
    GPTSimpleVectorIndex,
    GPTWeaviateIndex,
)
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.response.schema import Response

# TMP: refactor query config type
QUERY_CONFIG_TYPE = Union[Dict, QueryConfig]


# this is a map from type to outer index class
# we extract the type_to_struct and type_to_query
# fields from the index class
DEFAULT_INDEX_REGISTRY_MAP: Dict[IndexStructType, Type[BaseGPTIndex]] = {
    IndexStructType.TREE: GPTTreeIndex,
    IndexStructType.LIST: GPTListIndex,
    IndexStructType.KEYWORD_TABLE: GPTKeywordTableIndex,
    IndexStructType.SIMPLE_DICT: GPTSimpleVectorIndex,
    IndexStructType.DICT: GPTFaissIndex,
    IndexStructType.WEAVIATE: GPTWeaviateIndex,
    IndexStructType.PINECONE: GPTPineconeIndex,
    IndexStructType.QDRANT: GPTQdrantIndex,
    IndexStructType.CHROMA: GPTChromaIndex,
    IndexStructType.VECTOR_STORE: GPTVectorStoreIndex,
    IndexStructType.SQL: GPTSQLStructStoreIndex,
    IndexStructType.KG: GPTKnowledgeGraphIndex,
    IndexStructType.EMPTY: GPTEmptyIndex,
}



class CompositeIndexStruct(IndexStruct):
    all_index_structs: Dict[str, IndexStruct]
    root_id: str

class ComposableGraph:
    """Composable graph."""

    def __init__(
        self,
        index_struct: CompositeIndexStruct,
        docstore: DocumentStore,
        service_context: Optional[ServiceContext] = None,
    ) -> None:
        """Init params."""
        self._docstore = docstore
        self._index_struct = index_struct
        self._service_context = service_context or ServiceContext()

    @classmethod
    def from_indices(cls, all_indices: Dict[str, IndexStruct], root_id: str, docstores: Sequence[DocumentStore]):
        composite_index_struct = CompositeIndexStruct(
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
        # go over all the indices and create a registry
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
        # go over all the indices and create a registry
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
        # TODO: this is hardcoded for now, allow it to be specified by the user
        index_registry = _get_default_index_registry()
        docstore = DocumentStore.load_from_dict(
            result_dict["docstore"], index_registry.type_to_struct
        )
        index_struct = _safe_get_index_struct(docstore, result_dict["index_struct_id"])
        return cls(docstore, index_registry, index_struct, **kwargs)

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
            "index_struct_id": self._index_struct.get_doc_id(),
            "docstore": self._docstore.serialize_to_dict(),
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
