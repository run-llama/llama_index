"""Index registry."""

from typing import Dict

from gpt_index.data_structs.data_structs import PineconeIndexDict, SimpleIndexDict
from gpt_index.data_structs.data_structs_v2 import (
    KG,
    ChromaIndexDict,
    EmptyIndex,
    FaissIndexDict,
    IndexDict,
    IndexGraph,
    IndexList,
    KeywordTable,
    QdrantIndexDict,
    V2IndexStruct,
    WeaviateIndexDict,
)
from gpt_index.data_structs.node_v2 import Node
from gpt_index.data_structs.struct_type import IndexStructType
from gpt_index.data_structs.table_v2 import SQLStructTable
from gpt_index.indices.base import QueryMap

INDEX_STRUCT_TYPE_TO_INDEX_STRUCT_CLASS: Dict[IndexStructType, V2IndexStruct] = {
    IndexStructType.TREE: IndexGraph,
    IndexStructType.LIST: IndexList,
    IndexStructType.KEYWORD_TABLE: KeywordTable,
    IndexStructType.SIMPLE_DICT: SimpleIndexDict,
    IndexStructType.DICT: FaissIndexDict,
    IndexStructType.WEAVIATE: WeaviateIndexDict,
    IndexStructType.PINECONE: PineconeIndexDict,
    IndexStructType.QDRANT: QdrantIndexDict,
    IndexStructType.CHROMA: ChromaIndexDict,
    IndexStructType.VECTOR_STORE: IndexDict,
    IndexStructType.SQL: SQLStructTable,
    IndexStructType.KG: KG,
    IndexStructType.EMPTY: EmptyIndex,
}

# TODO: figure out how to avoid importing all indices while not centralizing all query map
INDEX_STRUT_TYPE_TO_QUERY_MAP: Dict[IndexStructType, QueryMap] = {
    index_type: index_cls.get_query_map()
    for index_type, index_cls in INDEX_STRUCT_TYPE_TO_INDEX_STRUCT_CLASS.items()
}
