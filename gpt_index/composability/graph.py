"""Composability graphs."""

import json
from typing import Any, Dict, List, Optional, Type, Union

from gpt_index.data_structs.data_structs import IndexStruct
from gpt_index.data_structs.struct_type import IndexStructType
from gpt_index.docstore import DocumentStore
from gpt_index.embeddings.base import BaseEmbedding
from gpt_index.embeddings.openai import OpenAIEmbedding
from gpt_index.indices.base import BaseGPTIndex
from gpt_index.indices.keyword_table.base import GPTKeywordTableIndex
from gpt_index.indices.knowledge_graph.base import GPTKnowledgeGraphIndex
from gpt_index.indices.list.base import GPTListIndex
from gpt_index.indices.prompt_helper import PromptHelper
from gpt_index.indices.query.query_runner import QueryRunner
from gpt_index.indices.query.schema import QueryBundle, QueryConfig
from gpt_index.indices.registry import IndexRegistry
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
}


def _get_default_index_registry() -> IndexRegistry:
    """Get default index registry."""
    index_registry = IndexRegistry()
    for index_type, index_class in DEFAULT_INDEX_REGISTRY_MAP.items():
        index_registry.type_to_struct[index_type] = index_class.index_struct_cls
        index_registry.type_to_query[index_type] = index_class.get_query_map()
    return index_registry


def _safe_get_index_struct(
    docstore: DocumentStore, index_struct_id: str
) -> IndexStruct:
    """Try get index struct."""
    index_struct = docstore.get_document(index_struct_id)
    if not isinstance(index_struct, IndexStruct):
        raise ValueError("Invalid `index_struct_id` - must be an IndexStruct")
    return index_struct


class ComposableGraph:
    """Composable graph."""

    def __init__(
        self,
        docstore: DocumentStore,
        index_registry: IndexRegistry,
        index_struct: IndexStruct,
        llm_predictor: Optional[LLMPredictor] = None,
        prompt_helper: Optional[PromptHelper] = None,
        embed_model: Optional[BaseEmbedding] = None,
        chunk_size_limit: Optional[int] = None,
    ) -> None:
        """Init params."""
        self._docstore = docstore
        self._index_registry = index_registry
        # this represents the "root" index struct
        self._index_struct = index_struct

        self._llm_predictor = llm_predictor or LLMPredictor()
        self._prompt_helper = prompt_helper or PromptHelper.from_llm_predictor(
            self._llm_predictor, chunk_size_limit=chunk_size_limit
        )
        self._embed_model = embed_model or OpenAIEmbedding()

    @classmethod
    def build_from_index(self, index: BaseGPTIndex) -> "ComposableGraph":
        """Build from index."""
        return ComposableGraph(
            index.docstore,
            index.index_registry,
            # this represents the "root" index struct
            index.index_struct,
            llm_predictor=index.llm_predictor,
            prompt_helper=index.prompt_helper,
            embed_model=index.embed_model,
        )

    def query(
        self,
        query_str: Union[str, QueryBundle],
        query_configs: Optional[List[QUERY_CONFIG_TYPE]] = None,
        llm_predictor: Optional[LLMPredictor] = None,
    ) -> Response:
        """Query the index."""
        # go over all the indices and create a registry
        llm_predictor = llm_predictor or self._llm_predictor
        query_runner = QueryRunner(
            llm_predictor,
            self._prompt_helper,
            self._embed_model,
            self._docstore,
            self._index_registry,
            query_configs=query_configs,
            recursive=True,
        )
        return query_runner.query(query_str, self._index_struct)

    def get_index(
        self, index_struct_id: str, index_cls: Type[BaseGPTIndex], **kwargs: Any
    ) -> BaseGPTIndex:
        """Get index."""
        index_struct = _safe_get_index_struct(self._docstore, index_struct_id)
        return index_cls(
            index_struct=index_struct,
            docstore=self._docstore,
            index_registry=self._index_registry,
            **kwargs
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
