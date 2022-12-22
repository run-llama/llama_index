"""Query map."""

from typing import Type

from gpt_index.indices.data_structs import IndexStructType
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.keyword_table.query import (
    GPTKeywordTableGPTQuery,
    GPTKeywordTableRAKEQuery,
    GPTKeywordTableSimpleQuery,
)
from gpt_index.indices.query.list.embedding_query import GPTListIndexEmbeddingQuery
from gpt_index.indices.query.list.query import GPTListIndexQuery
from gpt_index.indices.query.schema import QueryMode
from gpt_index.indices.query.tree.embedding_query import GPTTreeIndexEmbeddingQuery
from gpt_index.indices.query.tree.leaf_query import GPTTreeIndexLeafQuery
from gpt_index.indices.query.tree.retrieve_query import GPTTreeIndexRetQuery
from gpt_index.indices.query.tree.summarize_query import GPTTreeIndexSummarizeQuery
from gpt_index.indices.query.vector_store.faiss import GPTFaissIndexQuery

MODE_TO_QUERY_MAP_TREE = {
    QueryMode.DEFAULT: GPTTreeIndexLeafQuery,
    QueryMode.RETRIEVE: GPTTreeIndexRetQuery,
    QueryMode.EMBEDDING: GPTTreeIndexEmbeddingQuery,
    QueryMode.SUMMARIZE: GPTTreeIndexSummarizeQuery,
}

MODE_TO_QUERY_MAP_LIST = {
    QueryMode.DEFAULT: GPTListIndexQuery,
    QueryMode.EMBEDDING: GPTListIndexEmbeddingQuery,
}

MODE_TO_QUERY_MAP_KEYWORD_TABLE = {
    QueryMode.DEFAULT: GPTKeywordTableGPTQuery,
    QueryMode.SIMPLE: GPTKeywordTableRAKEQuery,
    QueryMode.RAKE: GPTKeywordTableSimpleQuery,
}

MODE_TO_QUERY_MAP_VECTOR = {
    QueryMode.DEFAULT: GPTFaissIndexQuery,
}


def get_query_cls(
    index_struct_type: IndexStructType, mode: QueryMode
) -> Type[BaseGPTIndexQuery]:
    """Get query class."""
    if index_struct_type == IndexStructType.TREE:
        return MODE_TO_QUERY_MAP_TREE[mode]
    elif index_struct_type == IndexStructType.LIST:
        return MODE_TO_QUERY_MAP_LIST[mode]
    elif index_struct_type == IndexStructType.KEYWORD_TABLE:
        return MODE_TO_QUERY_MAP_KEYWORD_TABLE[mode]
    elif index_struct_type == IndexStructType.DICT:
        return MODE_TO_QUERY_MAP_VECTOR[mode]
    else:
        raise ValueError(f"Invalid index_struct_type: {index_struct_type}")
