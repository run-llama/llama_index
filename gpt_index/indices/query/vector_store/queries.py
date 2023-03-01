"""Vector-store specific query classes."""


from typing import Any, Dict, Optional

from gpt_index.data_structs.data_structs import IndexDict
from gpt_index.indices.query.vector_store.base import GPTVectorStoreIndexQuery
from gpt_index.vector_stores import (
    ChromaVectorStore,
    FaissVectorStore,
    OpensearchVectorStore,
    PineconeVectorStore,
    QdrantVectorStore,
    SimpleVectorStore,
    WeaviateVectorStore,
)
from gpt_index.vector_stores.opensearch import OpensearchVectorClient


class GPTSimpleVectorIndexQuery(GPTVectorStoreIndexQuery):
    """GPT simple vector index query.

    Args:
        embed_model (Optional[BaseEmbedding]): embedding model
        similarity_top_k (int): number of top k results to return
        simple_vector_store_data_dict: (Optional[dict]): simple vector store data dict,

    """

    def __init__(
        self,
        index_struct: IndexDict,
        simple_vector_store_data_dict: Optional[Dict] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        # TODO: this is a temporary hack to allow composable
        # indices to work for simple vector stores
        # Our composability framework at the moment only allows for storage
        # of index_struct, not vector_store. Therefore in order to
        # allow simple vector indices to be composed, we need to "infer"
        # the vector store from the index struct.
        # NOTE: the next refactor would be to allow users to pass in
        # the vector store during query-time. However this is currently
        # not complete in our composability framework because the configs
        # are keyed on index type, not index id (which means that users
        # can't pass in distinct vector stores for different subindices).
        # NOTE: composability on top of other vector stores (pinecone/weaviate)
        # was already broken in this form.
        if simple_vector_store_data_dict is None:
            if len(index_struct.embeddings_dict) > 0:
                simple_vector_store_data_dict = {
                    "embedding_dict": index_struct.embeddings_dict,
                }
                vector_store = SimpleVectorStore(
                    simple_vector_store_data_dict=simple_vector_store_data_dict
                )
            else:
                raise ValueError("Vector store is required for vector store query.")
        else:
            vector_store = SimpleVectorStore(
                simple_vector_store_data_dict=simple_vector_store_data_dict
            )
        super().__init__(index_struct=index_struct, vector_store=vector_store, **kwargs)


class GPTFaissIndexQuery(GPTVectorStoreIndexQuery):
    """GPT faiss vector index query.

    Args:
        embed_model (Optional[BaseEmbedding]): embedding model
        similarity_top_k (int): number of top k results to return
        faiss_index (faiss.Index): A Faiss Index object (required). Note: the index
            will be reset during index construction.

    """

    def __init__(
        self,
        index_struct: IndexDict,
        faiss_index: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        if faiss_index is None:
            raise ValueError("faiss_index is required.")
        vector_store = FaissVectorStore(faiss_index)
        super().__init__(index_struct=index_struct, vector_store=vector_store, **kwargs)


class GPTPineconeIndexQuery(GPTVectorStoreIndexQuery):
    """GPT pinecone vector index query.

    Args:
        embed_model (Optional[BaseEmbedding]): embedding model
        similarity_top_k (int): number of top k results to return
        pinecone_index (Optional[pinecone.Index]): Pinecone index instance
        pinecone_kwargs (Optional[dict]): Pinecone index kwargs

    """

    def __init__(
        self,
        index_struct: IndexDict,
        pinecone_index: Optional[Any] = None,
        pinecone_kwargs: Optional[Dict] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        if pinecone_index is None and pinecone_kwargs is None:
            raise ValueError("pinecone_index or pinecone_kwargs is required.")
        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index, pinecone_kwargs=pinecone_kwargs
        )
        super().__init__(index_struct=index_struct, vector_store=vector_store, **kwargs)


class GPTWeaviateIndexQuery(GPTVectorStoreIndexQuery):
    """GPT Weaviate vector index query.

    Args:
        embed_model (Optional[BaseEmbedding]): embedding model
        similarity_top_k (int): number of top k results to return
        weaviate_client (Optional[Any]): Weaviate client instance
        class_prefix (Optional[str]): Weaviate class prefix

    """

    def __init__(
        self,
        index_struct: IndexDict,
        weaviate_client: Optional[Any] = None,
        class_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        if weaviate_client is None:
            raise ValueError("weaviate_client is required.")
        vector_store = WeaviateVectorStore(
            weaviate_client=weaviate_client, class_prefix=class_prefix
        )
        super().__init__(index_struct=index_struct, vector_store=vector_store, **kwargs)


class GPTQdrantIndexQuery(GPTVectorStoreIndexQuery):
    """GPT Qdrant vector index query.

    Args:
        embed_model (Optional[BaseEmbedding]): embedding model
        similarity_top_k (int): number of top k results to return
        client (Optional[Any]): QdrantClient instance from `qdrant-client` package
        collection_name: (Optional[str]): name of the Qdrant collection

    """

    def __init__(
        self,
        index_struct: IndexDict,
        client: Optional[Any] = None,
        collection_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        if client is None:
            raise ValueError("client is required.")
        if collection_name is None:
            raise ValueError("collection_name is required.")
        vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
        super().__init__(index_struct=index_struct, vector_store=vector_store, **kwargs)


class GPTChromaIndexQuery(GPTVectorStoreIndexQuery):
    """GPT Chroma vector index query.

    Args:
        text_qa_template (Optional[QuestionAnswerPrompt]): A Question-Answer Prompt
            (see :ref:`Prompt-Templates`).
        embed_model (Optional[BaseEmbedding]): Embedding model to use for
            embedding similarity.
        chroma_collection (Optional[Any]): Collection instance from `chromadb` package.

    """

    def __init__(
        self,
        index_struct: IndexDict,
        chroma_collection: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        if chroma_collection is None:
            raise ValueError("chroma_collection is required.")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        super().__init__(index_struct=index_struct, vector_store=vector_store, **kwargs)


class GPTOpensearchIndexQuery(GPTVectorStoreIndexQuery):
    """GPT Opensearch vector index query.

    Args:
        text_qa_template (Optional[QuestionAnswerPrompt]): A Question-Answer Prompt
            (see :ref:`Prompt-Templates`).
        embed_model (Optional[BaseEmbedding]): Embedding model to use for
            embedding similarity.
        client (Optional[OpensearchVectorClient]): Opensearch vector client.

    """

    def __init__(
        self,
        index_struct: IndexDict,
        client: Optional[OpensearchVectorClient] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        if client is None:
            raise ValueError("OpensearchVectorClient client is required.")
        vector_store = OpensearchVectorStore(client=client)
        super().__init__(index_struct=index_struct, vector_store=vector_store, **kwargs)
