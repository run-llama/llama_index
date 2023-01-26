"""IndexStructType class."""

from enum import Enum

from gpt_index.data_structs.data_structs import (
    IndexDict,
    IndexGraph,
    IndexList,
    IndexStruct,
    KeywordTable,
    Node,
    PineconeIndexStruct,
    SimpleIndexDict,
    WeaviateIndexStruct,
)
from gpt_index.data_structs.table import SQLStructTable


class IndexStructType(str, Enum):
    """Index struct type. Identifier for a "type" of index.

    Attributes:
        TREE ("tree"): Tree index. See :ref:`Ref-Indices-Tree` for tree indices.
        LIST ("list"): List index. See :ref:`Ref-Indices-List` for list indices.
        KEYWORD_TABLE ("keyword_table"): Keyword table index. See
            :ref:`Ref-Indices-Table`
            for keyword table indices.
        DICT ("dict"): Faiss Vector Store Index. See :ref:`Ref-Indices-VectorStore`
            for more information on the Faiss vector store index.
        SIMPLE_DICT ("simple_dict"): Simple Vector Store Index. See
            :ref:`Ref-Indices-VectorStore`
            for more information on the simple vector store index.
        WEAVIATE ("weaviate"): Weaviate Vector Store Index.
            See :ref:`Ref-Indices-VectorStore`
            for more information on the Weaviate vector store index.
        PINECONE ("pinecone"): Pinecone Vector Store Index.
            See :ref:`Ref-Indices-VectorStore`
            for more information on the Pinecone vector store index.

        SQL ("SQL"): SQL Structured Store Index.
            See :ref:`Ref-Indices-StructStore`
            for more information on the SQL vector store index.

    """

    # TODO: refactor so these are properties on the base class

    NODE = "node"
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
    # for pinecone index
    PINECONE = "pinecone"
    # for SQL index
    SQL = "sql"

    def get_index_struct_cls(self) -> type:
        """Get index struct class."""
        if self == IndexStructType.NODE:
            return Node
        elif self == IndexStructType.TREE:
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
        elif self == IndexStructType.PINECONE:
            return PineconeIndexStruct
        elif self == IndexStructType.SQL:
            return SQLStructTable
        else:
            raise ValueError("Invalid index struct type.")

    @classmethod
    def from_index_struct(cls, index_struct: IndexStruct) -> "IndexStructType":
        """Get index enum from index struct class."""
        if isinstance(index_struct, Node):
            return cls.NODE
        elif isinstance(index_struct, IndexGraph):
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
        elif isinstance(index_struct, PineconeIndexStruct):
            return cls.PINECONE
        elif isinstance(index_struct, SQLStructTable):
            return cls.SQL
        else:
            raise ValueError("Invalid index struct type.")
