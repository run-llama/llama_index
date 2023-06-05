"""Index registry."""

from typing import Dict, Type

from llama_index.data_structs.struct_type import IndexStructType
from llama_index.indices.base import BaseGPTIndex
from llama_index.indices.document_summary.base import GPTDocumentSummaryIndex
from llama_index.indices.empty.base import GPTEmptyIndex
from llama_index.indices.keyword_table.base import GPTKeywordTableIndex
from llama_index.indices.knowledge_graph.base import GPTKnowledgeGraphIndex
from llama_index.indices.list.base import ListIndex
from llama_index.indices.struct_store.pandas import GPTPandasIndex
from llama_index.indices.struct_store.sql import GPTSQLStructStoreIndex
from llama_index.indices.tree.base import GPTTreeIndex
from llama_index.indices.vector_store.base import GPTVectorStoreIndex

INDEX_STRUCT_TYPE_TO_INDEX_CLASS: Dict[IndexStructType, Type[BaseGPTIndex]] = {
    IndexStructType.TREE: GPTTreeIndex,
    IndexStructType.LIST: ListIndex,
    IndexStructType.KEYWORD_TABLE: GPTKeywordTableIndex,
    IndexStructType.VECTOR_STORE: GPTVectorStoreIndex,
    IndexStructType.SQL: GPTSQLStructStoreIndex,
    IndexStructType.PANDAS: GPTPandasIndex,
    IndexStructType.KG: GPTKnowledgeGraphIndex,
    IndexStructType.EMPTY: GPTEmptyIndex,
    IndexStructType.DOCUMENT_SUMMARY: GPTDocumentSummaryIndex,
}
