"""Index registry."""

from typing import Dict, Type

from llama_index.data_structs.data_structs import (
    KG,
    EmptyIndex,
    IndexDict,
    IndexGraph,
    IndexList,
    KeywordTable,
    IndexStruct,
)
from llama_index.data_structs.struct_type import IndexStructType
from llama_index.data_structs.table import PandasStructTable, SQLStructTable
from llama_index.indices.base import BaseGPTIndex
from llama_index.indices.empty.base import GPTEmptyIndex
from llama_index.indices.keyword_table.base import GPTKeywordTableIndex
from llama_index.indices.knowledge_graph.base import GPTKnowledgeGraphIndex
from llama_index.indices.list.base import GPTListIndex
from llama_index.indices.struct_store.pandas import GPTPandasIndex
from llama_index.indices.struct_store.sql import GPTSQLStructStoreIndex
from llama_index.indices.tree.base import GPTTreeIndex
from llama_index.indices.vector_store.base import GPTVectorStoreIndex

INDEX_STRUCT_TYPE_TO_INDEX_STRUCT_CLASS: Dict[IndexStructType, Type[IndexStruct]] = {
    IndexStructType.TREE: IndexGraph,
    IndexStructType.LIST: IndexList,
    IndexStructType.KEYWORD_TABLE: KeywordTable,
    IndexStructType.VECTOR_STORE: IndexDict,
    IndexStructType.SQL: SQLStructTable,
    IndexStructType.PANDAS: PandasStructTable,
    IndexStructType.KG: KG,
    IndexStructType.EMPTY: EmptyIndex,
}


INDEX_STRUCT_TYPE_TO_INDEX_CLASS: Dict[IndexStructType, Type[BaseGPTIndex]] = {
    IndexStructType.TREE: GPTTreeIndex,
    IndexStructType.LIST: GPTListIndex,
    IndexStructType.KEYWORD_TABLE: GPTKeywordTableIndex,
    IndexStructType.VECTOR_STORE: GPTVectorStoreIndex,
    IndexStructType.SQL: GPTSQLStructStoreIndex,
    IndexStructType.PANDAS: GPTPandasIndex,
    IndexStructType.KG: GPTKnowledgeGraphIndex,
    IndexStructType.EMPTY: GPTEmptyIndex,
}
