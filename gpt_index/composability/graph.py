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
from gpt_index.indices.list.base import GPTListIndex
from gpt_index.indices.prompt_helper import PromptHelper
from gpt_index.indices.query.query_runner import QueryRunner
from gpt_index.indices.query.schema import QueryConfig
from gpt_index.indices.registry import IndexRegistry
from gpt_index.indices.struct_store.sql import GPTSQLStructStoreIndex
from gpt_index.indices.tree.base import GPTTreeIndex
from gpt_index.indices.vector_store.faiss import GPTFaissIndex
from gpt_index.indices.vector_store.pinecone import GPTPineconeIndex
from gpt_index.indices.vector_store.simple import GPTSimpleVectorIndex
from gpt_index.indices.vector_store.weaviate import GPTWeaviateIndex
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
    IndexStructType.DICT: GPTFaissIndex,
    IndexStructType.SIMPLE_DICT: GPTSimpleVectorIndex,
    IndexStructType.WEAVIATE: GPTWeaviateIndex,
    IndexStructType.PINECONE: GPTPineconeIndex,
    IndexStructType.SQL: GPTSQLStructStoreIndex,
}


def _get_default_index_registry() -> IndexRegistry:
    """Get default index registry."""
    index_registry = IndexRegistry()
    for index_type, index_class in DEFAULT_INDEX_REGISTRY_MAP.items():
        index_registry.type_to_struct[index_type] = index_class.index_struct_cls
        index_registry.type_to_query[index_type] = index_class.get_query_map()
    return index_registry


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
        query_str: str,
        query_configs: Optional[List[QUERY_CONFIG_TYPE]],
        verbose: bool = False,
    ) -> Response:
        """Query the index."""
        # go over all the indices and create a registry
        query_runner = QueryRunner(
            self._llm_predictor,
            self._prompt_helper,
            self._embed_model,
            self._docstore,
            self._index_registry,
            query_configs=query_configs,
            verbose=verbose,
            recursive=True,
        )
        return query_runner.query(query_str, self._index_struct)

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
            result_dict = json.load(f)
            # TODO: this is hardcoded for now, allow it to be specified by the user
            index_registry = _get_default_index_registry()
            docstore = DocumentStore.load_from_dict(
                result_dict["docstore"], index_registry.type_to_struct
            )
            index_struct = docstore.get_document(result_dict["index_struct_id"])
            if not isinstance(index_struct, IndexStruct):
                raise ValueError("Invalid `index_struct_id` - must be an IndexStruct")
            return cls(docstore, index_registry, index_struct, **kwargs)

    def save_to_disk(self, save_path: str, **save_kwargs: Any) -> None:
        """Save to file.

        This method stores the index into a JSON file stored on disk.

        Args:
            save_path (str): The save_path of the file.

        """
        out_dict: Dict[str, Any] = {
            "index_struct_id": self._index_struct.get_doc_id(),
            "docstore": self._docstore.serialize_to_dict(),
        }
        with open(save_path, "w") as f:
            json.dump(out_dict, f)
