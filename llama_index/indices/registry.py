"""Index registry."""

from typing import Dict, Type

from llama_index.data_structs.struct_type import IndexStructType
from llama_index.indices.base import BaseIndex
from llama_index.indices.document_summary.base import GPTDocumentSummaryIndex
from llama_index.indices.empty.base import GPTEmptyIndex
from llama_index.indices.keyword_table.base import KeywordTableIndex
from llama_index.indices.knowledge_graph.base import GPTKnowledgeGraphIndex
from llama_index.indices.list.base import ListIndex
from llama_index.indices.struct_store.pandas import PandasIndex
from llama_index.indices.struct_store.sql import SQLStructStoreIndex
from llama_index.indices.tree.base import TreeIndex
from llama_index.indices.vector_store.base import VectorStoreIndex

INDEX_STRUCT_TYPE_TO_INDEX_CLASS: Dict[IndexStructType, Type[BaseIndex]] = {
    IndexStructType.TREE: TreeIndex,
    IndexStructType.LIST: ListIndex,
    IndexStructType.KEYWORD_TABLE: KeywordTableIndex,
    IndexStructType.VECTOR_STORE: VectorStoreIndex,
    IndexStructType.SQL: SQLStructStoreIndex,
    IndexStructType.PANDAS: PandasIndex,
    IndexStructType.KG: GPTKnowledgeGraphIndex,
    IndexStructType.EMPTY: GPTEmptyIndex,
    IndexStructType.DOCUMENT_SUMMARY: GPTDocumentSummaryIndex,
}
