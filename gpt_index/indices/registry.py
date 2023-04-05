"""Index registry."""

from typing import Any, Dict, Type

from gpt_index.constants import DATA_KEY, TYPE_KEY
from gpt_index.data_structs.data_structs_v2 import (
    KG,
    ChromaIndexDict,
    OpensearchIndexDict,
    CompositeIndex,
    EmptyIndex,
    FaissIndexDict,
    IndexDict,
    IndexGraph,
    IndexList,
    KeywordTable,
    PineconeIndexDict,
    QdrantIndexDict,
    SimpleIndexDict,
    V2IndexStruct,
    WeaviateIndexDict,
    ChatGPTRetrievalPluginIndexDict,
)
from gpt_index.data_structs.struct_type import IndexStructType
from gpt_index.data_structs.table_v2 import PandasStructTable, SQLStructTable
from gpt_index.indices.base import BaseGPTIndex, QueryMap
from gpt_index.indices.empty.base import GPTEmptyIndex
from gpt_index.indices.keyword_table.base import GPTKeywordTableIndex
from gpt_index.indices.knowledge_graph.base import GPTKnowledgeGraphIndex
from gpt_index.indices.list.base import GPTListIndex
from gpt_index.indices.struct_store.pandas import GPTPandasIndex
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
    ChatGPTRetrievalPluginIndex,
    GPTOpensearchIndex,
)

INDEX_STRUCT_TYPE_TO_INDEX_STRUCT_CLASS: Dict[IndexStructType, Type[V2IndexStruct]] = {
    IndexStructType.TREE: IndexGraph,
    IndexStructType.LIST: IndexList,
    IndexStructType.KEYWORD_TABLE: KeywordTable,
    IndexStructType.SIMPLE_DICT: SimpleIndexDict,
    IndexStructType.DICT: FaissIndexDict,
    IndexStructType.WEAVIATE: WeaviateIndexDict,
    IndexStructType.PINECONE: PineconeIndexDict,
    IndexStructType.QDRANT: QdrantIndexDict,
    IndexStructType.CHROMA: ChromaIndexDict,
    IndexStructType.OPENSEARCH: OpensearchIndexDict,
    IndexStructType.VECTOR_STORE: IndexDict,
    IndexStructType.SQL: SQLStructTable,
    IndexStructType.PANDAS: PandasStructTable,
    IndexStructType.KG: KG,
    IndexStructType.EMPTY: EmptyIndex,
    IndexStructType.COMPOSITE: CompositeIndex,
    IndexStructType.CHATGPT_RETRIEVAL_PLUGIN: ChatGPTRetrievalPluginIndexDict,
}


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
    IndexStructType.PANDAS: GPTPandasIndex,
    IndexStructType.KG: GPTKnowledgeGraphIndex,
    IndexStructType.EMPTY: GPTEmptyIndex,
    IndexStructType.CHATGPT_RETRIEVAL_PLUGIN: ChatGPTRetrievalPluginIndex,
    IndexStructType.OPENSEARCH: GPTOpensearchIndex,
}


INDEX_STRUT_TYPE_TO_QUERY_MAP: Dict[IndexStructType, QueryMap] = {
    index_type: index_cls.get_query_map()
    for index_type, index_cls in INDEX_STRUCT_TYPE_TO_INDEX_CLASS.items()
}


def load_index_struct_from_dict(struct_dict: Dict[str, Any]) -> "V2IndexStruct":
    type = struct_dict[TYPE_KEY]
    data_dict = struct_dict[DATA_KEY]

    cls = INDEX_STRUCT_TYPE_TO_INDEX_STRUCT_CLASS[type]

    if type == IndexStructType.COMPOSITE:
        struct_dicts: Dict[str, Any] = data_dict["all_index_structs"]
        root_id = data_dict["root_id"]
        all_index_structs = {
            id_: load_index_struct_from_dict(struct_dict)
            for id_, struct_dict in struct_dicts.items()
        }
        return CompositeIndex(all_index_structs=all_index_structs, root_id=root_id)
    else:
        return cls.from_dict(data_dict)
