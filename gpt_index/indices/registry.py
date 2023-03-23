"""Index registry."""

from typing import Any, Dict, Type

from gpt_index.data_structs.struct_type import IndexStructType
from gpt_index.indices.base import BaseGPTIndex, QueryMap
from gpt_index.indices.empty.base import GPTEmptyIndex
from gpt_index.indices.keyword_table.base import GPTKeywordTableIndex
from gpt_index.indices.knowledge_graph.base import GPTKnowledgeGraphIndex
from gpt_index.indices.list.base import GPTListIndex
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

INDEX_STRUCT_TYPE_TO_INDEX_CLASS: Dict[IndexStructType, Type[BaseGPTIndex]] = {
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

INDEX_STRUCT_TYPE_TO_INDEX_STRUCT_CLASS: Dict[IndexStructType, Any] = {
    index_type: index_cls.index_struct_cls
    for index_type, index_cls in INDEX_STRUCT_TYPE_TO_INDEX_CLASS.items()
}

INDEX_STRUT_TYPE_TO_QUERY_MAP: Dict[IndexStructType, QueryMap] = {
    index_type: index_cls.get_query_map()
    for index_type, index_cls in INDEX_STRUCT_TYPE_TO_INDEX_CLASS.items()
}
