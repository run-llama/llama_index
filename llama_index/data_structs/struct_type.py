"""IndexStructType class."""

from enum import Enum


class IndexStructType(str, Enum):
    """Index struct type. Identifier for a "type" of index.

    Attributes:
        TREE ("tree"): Tree index. See :ref:`Ref-Indices-Tree` for tree indices.
        LIST ("list"): Summary index. See :ref:`Ref-Indices-List` for summary indices.
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
        DEEPLAKE ("deeplake"): DeepLake Vector Store Index.
            See :ref:`Ref-Indices-VectorStore`
            for more information on the Pinecone vector store index.
        QDRANT ("qdrant"): Qdrant Vector Store Index.
            See :ref:`Ref-Indices-VectorStore`
            for more information on the Qdrant vector store index.
        LANCEDB ("lancedb"): LanceDB Vector Store Index
            See :ref:`Ref-Indices-VectorStore`
            for more information on the LanceDB vector store index.
        MILVUS ("milvus"): Milvus Vector Store Index.
            See :ref:`Ref-Indices-VectorStore`
            for more information on the Milvus vector store index.
        CHROMA ("chroma"): Chroma Vector Store Index.
            See :ref:`Ref-Indices-VectorStore`
            for more information on the Chroma vector store index.
        OPENSEARCH ("opensearch"): Opensearch Vector Store Index.
            See :ref:`Ref-Indices-VectorStore`
            for more information on the Opensearch vector store index.
        MYSCALE ("myscale"): MyScale Vector Store Index.
            See :ref:`Ref-Indices-VectorStore`
            for more information on the MyScale vector store index.
        EPSILLA ("epsilla"): Epsilla Vector Store Index.
            See :ref:`Ref-Indices-VectorStore`
            for more information on the Epsilla vector store index.
        CHATGPT_RETRIEVAL_PLUGIN ("chatgpt_retrieval_plugin"): ChatGPT
            retrieval plugin index.
        SQL ("SQL"): SQL Structured Store Index.
            See :ref:`Ref-Indices-StructStore`
            for more information on the SQL vector store index.
        DASHVECTOR ("dashvector"): DashVector Vector Store Index.
            See :ref:`Ref-Indices-VectorStore`
            for more information on the Dashvecotor vector store index.
        KG ("kg"): Knowledge Graph index.
            See :ref:`Ref-Indices-Knowledge-Graph` for KG indices.
        DOCUMENT_SUMMARY ("document_summary"): Document Summary Index.
            See :ref:`Ref-Indices-Document-Summary` for Summary Indices.

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
    LANCEDB = "lancedb"
    MILVUS = "milvus"
    CHROMA = "chroma"
    MYSCALE = "myscale"
    VECTOR_STORE = "vector_store"
    OPENSEARCH = "opensearch"
    DASHVECTOR = "dashvector"
    CHATGPT_RETRIEVAL_PLUGIN = "chatgpt_retrieval_plugin"
    DEEPLAKE = "deeplake"
    EPSILLA = "epsilla"
    # multimodal
    MULTIMODAL_VECTOR_STORE = "multimodal"
    # for SQL index
    SQL = "sql"
    # for KG index
    KG = "kg"
    SIMPLE_KG = "simple_kg"
    NEBULAGRAPH = "nebulagraph"
    FALKORDB = "falkordb"

    # EMPTY
    EMPTY = "empty"
    COMPOSITE = "composite"

    PANDAS = "pandas"

    DOCUMENT_SUMMARY = "document_summary"

    # Managed
    VECTARA = "vectara"
    ZILLIZ_CLOUD_PIPELINE = "zilliz_cloud_pipeline"
