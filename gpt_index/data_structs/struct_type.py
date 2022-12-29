"""Index struct type."""

from enum import Enum
from gpt_index.data_structs.data_structs import (
    IndexGraph, IndexList, KeywordTable,
    IndexDict,
    SimpleIndexDict,
    WeaviateIndexStruct,
    IndexStruct
)


class IndexStructType(str, Enum):
    """Index struct type."""

    TREE = "tree"
    LIST = "list"
    KEYWORD_TABLE = "keyword_table"
    # for Faiss
    # TODO: rename
    DICT = "dict"
    # for simple embedding index
    SIMPLE_DICT = "simple_dict"
    # for weaviate index
    WEAVIATE = "weaviate"

    def get_index_struct_cls(self) -> type:
        """Get index struct class."""
        if self == IndexStructType.TREE:
            return IndexGraph
        elif self == IndexStructType.LIST:
            return IndexList
        elif self == IndexStructType.KEYWORD_TABLE:
            return KeywordTable
        elif self == IndexStructType.DICT:
            return IndexDict
        elif self == IndexStructType.SIMPLE_DICT:
            return SimpleIndexDict
        elif self == IndexStructType.WEAVIATE:
            return WeaviateIndexStruct
        else:
            raise ValueError("Invalid index struct type.")

    @classmethod
    def from_index_struct(cls, index_struct: IndexStruct) -> "IndexStructType":
        """Get index enum from index struct class."""
        if isinstance(index_struct, IndexGraph):
            return cls.TREE
        elif isinstance(index_struct, IndexList):
            return cls.LIST
        elif isinstance(index_struct, KeywordTable):
            return cls.KEYWORD_TABLE
        elif isinstance(index_struct, IndexDict):
            return cls.DICT
        elif isinstance(index_struct, SimpleIndexDict):
            return cls.SIMPLE_DICT
        elif isinstance(index_struct, WeaviateIndexStruct):
            return cls.WEAVIATE
        else:
            raise ValueError("Invalid index struct type.")