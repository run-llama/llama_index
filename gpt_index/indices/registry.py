"""Index registry."""

from typing import Any, Dict

from gpt_index.data_structs.data_structs import PineconeIndexDict, SimpleIndexDict
from gpt_index.data_structs.data_structs_v2 import (
    DATA_KEY,
    KG,
    TYPE_KEY,
    ChromaIndexDict,
    CompositeIndex,
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
    IndexStructType.COMPOSITE: CompositeIndex
}

# TODO: figure out how to avoid importing all indices while not centralizing all query map
INDEX_STRUT_TYPE_TO_QUERY_MAP: Dict[IndexStructType, QueryMap] = {
    index_type: index_cls.get_query_map()
    for index_type, index_cls in INDEX_STRUCT_TYPE_TO_INDEX_STRUCT_CLASS.items()
}



@classmethod
def load_index_struct_from_dict(struct_dict: Dict[str, Any]) -> "V2IndexStruct":
    type = struct_dict[TYPE_KEY]
    data_dict = struct_dict[DATA_KEY]
    
    cls = INDEX_STRUCT_TYPE_TO_INDEX_STRUCT_CLASS[type]
    
    if type == IndexStructType.COMPOSITE:
        struct_dicts: Dict[str, Any] = data_dict['all_index_structs']
        root_id = data_dict['root_id']
        all_index_structs = {
            id_: load_index_struct_from_dict(struct_dict)
            for id_, struct_dict in struct_dicts.items()
        }
        return CompositeIndex(all_index_structs=all_index_structs, root_id=root_id)
    else:
        return cls.from_dict(data_dict)
    