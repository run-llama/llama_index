"""Index registry."""

from typing import Dict, Type

from llama_index.data_structs.struct_type import IndexStructType
from llama_index.indices.base import BaseIndex
from llama_index.indices.document_summary.base import DocumentSummaryIndex
from llama_index.indices.empty.base import EmptyIndex
from llama_index.indices.keyword_table.base import KeywordTableIndex
from llama_index.indices.knowledge_graph.base import KnowledgeGraphIndex
from llama_index.indices.list.base import SummaryIndex
from llama_index.indices.struct_store.pandas import PandasIndex
from llama_index.indices.struct_store.sql import SQLStructStoreIndex
from llama_index.indices.tree.base import TreeIndex
from llama_index.indices.vector_store.base import VectorStoreIndex

INDEX_STRUCT_TYPE_TO_INDEX_CLASS: Dict[IndexStructType, Type[BaseIndex]] = {
    IndexStructType.TREE: TreeIndex,
    IndexStructType.LIST: SummaryIndex,
    IndexStructType.KEYWORD_TABLE: KeywordTableIndex,
    IndexStructType.VECTOR_STORE: VectorStoreIndex,
    IndexStructType.SQL: SQLStructStoreIndex,
    IndexStructType.PANDAS: PandasIndex,
    IndexStructType.KG: KnowledgeGraphIndex,
    IndexStructType.EMPTY: EmptyIndex,
    IndexStructType.DOCUMENT_SUMMARY: DocumentSummaryIndex,
}
