"""IndexStructType class."""

from enum import Enum


class IndexStructType(str, Enum):
    """Index struct type. Identifier for a "type" of index.

    Attributes:
        TREE ("tree"): Tree index. See :ref:`Ref-Indices-Tree` for tree indices.
        LIST ("list"): List index. See :ref:`Ref-Indices-List` for list indices.
        KEYWORD_TABLE ("keyword_table"): Keyword table index. See
            :ref:`Ref-Indices-Table`
            for keyword table indices.
        DICT ("dict"): Faiss Vector Store Index. See
            :ref:`Ref-Indices-VectorStore`
            for more information on the faiss vector store index.
        SIMPLE_DICT ("simple_dict"): Simple Vector Store Index. See
            :ref:`Ref-Indices-VectorStore`
            for more information on the simple vector store index.
        WEAVIATE ("weaviate"): Weaviate Vector Store Index.
            See :ref:`Ref-Indices-VectorStore`
            for more information on the Weaviate vector store index.
        PINECONE ("pinecone"): Pinecone Vector Store Index.
            See :ref:`Ref-Indices-VectorStore`
            for more information on the Pinecone vector store index.
        QDRANT ("qdrant"): Qdrant Vector Store Index.
            See :ref:`Ref-Indices-VectorStore`
            for more information on the Qdrant vector store index.
        CHROMA ("chroma"): Chroma Vector Store Index.
            See :ref:`Ref-Indices-VectorStore`
            for more information on the Chroma vector store index.
        OPENSEARCH ("opensearch"): Opensearch Vector Store Index.
            See :ref:`Ref-Indices-VectorStore`
            for more information on the Opensearch vector store index.
        SQL ("SQL"): SQL Structured Store Index.
            See :ref:`Ref-Indices-StructStore`
            for more information on the SQL vector store index.
        KG ("kg"): Knowledge Graph index.
            See :ref:`Ref-Indices-Knowledge-Graph` for KG indices.

    """

    # TODO: refactor so these are properties on the base class

    NODE = "node"
    TREE = "tree"
    LIST = "list"
    KEYWORD_TABLE = "keyword_table"

    # faiss
    DICT = "dict"
    # simple
    SIMPLE_DICT = "simple_dict"
    WEAVIATE = "weaviate"
    PINECONE = "pinecone"
    QDRANT = "qdrant"
    CHROMA = "chroma"
    VECTOR_STORE = "vector_store"
    OPENSEARCH = "opensearch"

    # for SQL index
    SQL = "sql"
    # for KG index
    KG = "kg"
