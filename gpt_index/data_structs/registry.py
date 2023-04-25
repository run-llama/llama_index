"""Index registry."""

from typing import Dict, Type

from gpt_index.data_structs.data_structs_v2 import (
    KG,
    ChatGPTRetrievalPluginIndexDict,
    ChromaIndexDict,
    CompositeIndex,
    DeepLakeIndexDict,
    EmptyIndex,
    FaissIndexDict,
    IndexDict,
    IndexGraph,
    IndexList,
    KeywordTable,
    MilvusIndexDict,
    MyScaleIndexDict,
    OpensearchIndexDict,
    PineconeIndexDict,
    QdrantIndexDict,
    SimpleIndexDict,
    V2IndexStruct,
    WeaviateIndexDict,
)
from gpt_index.data_structs.struct_type import IndexStructType
from gpt_index.data_structs.table_v2 import PandasStructTable, SQLStructTable


INDEX_STRUCT_TYPE_TO_INDEX_STRUCT_CLASS: Dict[IndexStructType, Type[V2IndexStruct]] = {
    IndexStructType.TREE: IndexGraph,
    IndexStructType.LIST: IndexList,
    IndexStructType.KEYWORD_TABLE: KeywordTable,
    IndexStructType.SIMPLE_DICT: SimpleIndexDict,
    IndexStructType.DICT: FaissIndexDict,
    IndexStructType.WEAVIATE: WeaviateIndexDict,
    IndexStructType.PINECONE: PineconeIndexDict,
    IndexStructType.QDRANT: QdrantIndexDict,
    IndexStructType.MILVUS: MilvusIndexDict,
    IndexStructType.CHROMA: ChromaIndexDict,
    IndexStructType.OPENSEARCH: OpensearchIndexDict,
    IndexStructType.VECTOR_STORE: IndexDict,
    IndexStructType.SQL: SQLStructTable,
    IndexStructType.PANDAS: PandasStructTable,
    IndexStructType.KG: KG,
    IndexStructType.EMPTY: EmptyIndex,
    IndexStructType.COMPOSITE: CompositeIndex,
    IndexStructType.CHATGPT_RETRIEVAL_PLUGIN: ChatGPTRetrievalPluginIndexDict,
    IndexStructType.DEEPLAKE: DeepLakeIndexDict,
    IndexStructType.MYSCALE: MyScaleIndexDict,
}
