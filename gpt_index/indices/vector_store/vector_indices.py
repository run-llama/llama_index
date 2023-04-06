"""Deprecated vector store indices."""

from typing import Any, Dict, Optional, Sequence, Type

from requests.adapters import Retry

from gpt_index.data_structs.data_structs_v2 import (
    ChatGPTRetrievalPluginIndexDict,
    ChromaIndexDict,
    FaissIndexDict,
    IndexDict,
    OpensearchIndexDict,
    PineconeIndexDict,
    QdrantIndexDict,
    SimpleIndexDict,
    WeaviateIndexDict,
)
from gpt_index.data_structs.node_v2 import Node
from gpt_index.indices.base import BaseGPTIndex
from gpt_index.indices.service_context import ServiceContext
from gpt_index.indices.vector_store.base import GPTVectorStoreIndex
from gpt_index.vector_stores import (
    ChatGPTRetrievalPluginClient,
    ChromaVectorStore,
    FaissVectorStore,
    PineconeVectorStore,
    QdrantVectorStore,
    SimpleVectorStore,
    WeaviateVectorStore,
)
from gpt_index.vector_stores.opensearch import (
    OpensearchVectorClient,
    OpensearchVectorStore,
)


class GPTSimpleVectorIndex(GPTVectorStoreIndex):
    """GPT Simple Vector Index.

    The GPTSimpleVectorIndex is a data structure where nodes are keyed by
    embeddings, and those embeddings are stored within a simple dictionary.
    During index construction, the document texts are chunked up,
    converted to nodes with text; they are then encoded in
    document embeddings stored within the dict.

    During query time, the index uses the dict to query for the top
    k most similar nodes, and synthesizes an answer from the
    retrieved nodes.

    Args:
        service_context (ServiceContext): Service context container (contains
            components like LLMPredictor, PromptHelper, etc.).

    """

    index_struct_cls: Type[IndexDict] = SimpleIndexDict

    def __init__(
        self,
        nodes: Optional[Sequence[Node]] = None,
        index_struct: Optional[IndexDict] = None,
        service_context: Optional[ServiceContext] = None,
        vector_store: Optional[SimpleVectorStore] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            service_context=service_context,
            vector_store=vector_store,
            **kwargs,
        )


class GPTFaissIndex(GPTVectorStoreIndex):
    """GPT Faiss Index.

    The GPTFaissIndex is a data structure where nodes are keyed by
    embeddings, and those embeddings are stored within a Faiss index.
    During index construction, the document texts are chunked up,
    converted to nodes with text; they are then encoded in
    document embeddings stored within Faiss.

    During query time, the index uses Faiss to query for the top
    k most similar nodes, and synthesizes an answer from the
    retrieved nodes.

    Args:
        faiss_index (faiss.Index): A Faiss Index object (required). Note: the index
            will be reset during index construction.
        service_context (ServiceContext): Service context container (contains
            components like LLMPredictor, PromptHelper, etc.).
    """

    index_struct_cls: Type[IndexDict] = FaissIndexDict

    def __init__(
        self,
        nodes: Optional[Sequence[Node]] = None,
        service_context: Optional[ServiceContext] = None,
        faiss_index: Optional[Any] = None,
        index_struct: Optional[IndexDict] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        if faiss_index is None:
            raise ValueError("faiss_index is required.")
        vector_store = FaissVectorStore(faiss_index)

        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            service_context=service_context,
            vector_store=vector_store,
            **kwargs,
        )

    @classmethod
    def load_from_disk(
        cls, save_path: str, faiss_index_save_path: Optional[str] = None, **kwargs: Any
    ) -> "BaseGPTIndex":
        """Load index from disk.

        This method loads the index from a JSON file stored on disk. The index data
        structure itself is preserved completely. If the index is defined over
        subindices, those subindices will also be preserved (and subindices of
        those subindices, etc.).
        In GPTFaissIndex, we allow user to specify an additional
        `faiss_index_save_path` to load faiss index from a file - that
        way, the user does not have to recreate the faiss index outside
        of this class.

        Args:
            save_path (str): The save_path of the file.
           faiss_index_save_path (Optional[str]): The save_path of the
                Faiss index file. If not specified, the Faiss index
                will not be saved to disk.
            **kwargs: Additional kwargs to pass to the index constructor.

        Returns:
            BaseGPTIndex: The loaded index.
        """
        if faiss_index_save_path is not None:
            import faiss

            faiss_index = faiss.read_index(faiss_index_save_path)
            return super().load_from_disk(save_path, faiss_index=faiss_index, **kwargs)
        else:
            return super().load_from_disk(save_path, **kwargs)

    def save_to_disk(
        self,
        save_path: str,
        encoding: str = "ascii",
        faiss_index_save_path: Optional[str] = None,
        **save_kwargs: Any,
    ) -> None:
        """Save to file.

        This method stores the index into a JSON file stored on disk.
        In GPTFaissIndex, we allow user to specify an additional
        `faiss_index_save_path` to save the faiss index to a file - that
        way, the user can pass in the same argument in
        `GPTFaissIndex.load_from_disk` without having to recreate
        the Faiss index outside of this class.

        Args:
            save_path (str): The save_path of the file.
            encoding (str): The encoding to use when saving the file.
            faiss_index_save_path (Optional[str]): The save_path of the
                Faiss index file. If not specified, the Faiss index
                will not be saved to disk.
        """
        super().save_to_disk(save_path, encoding=encoding, **save_kwargs)

        if faiss_index_save_path is not None:
            import faiss

            faiss.write_index(self._vector_store.client, faiss_index_save_path)


class GPTPineconeIndex(GPTVectorStoreIndex):
    """GPT Pinecone Index.

    The GPTPineconeIndex is a data structure where nodes are keyed by
    embeddings, and those embeddings are stored within a Pinecone index.
    During index construction, the document texts are chunked up,
    converted to nodes with text; they are then encoded in
    document embeddings stored within Pinecone.

    During query time, the index uses Pinecone to query for the top
    k most similar nodes, and synthesizes an answer from the
    retrieved nodes.

    Args:
        service_context (ServiceContext): Service context container (contains
            components like LLMPredictor, PromptHelper, etc.).
    """

    index_struct_cls: Type[IndexDict] = PineconeIndexDict

    def __init__(
        self,
        nodes: Optional[Sequence[Node]] = None,
        pinecone_index: Optional[Any] = None,
        index_name: Optional[str] = None,
        environment: Optional[str] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        pinecone_kwargs: Optional[Dict] = None,
        insert_kwargs: Optional[Dict] = None,
        query_kwargs: Optional[Dict] = None,
        delete_kwargs: Optional[Dict] = None,
        index_struct: Optional[IndexDict] = None,
        service_context: Optional[ServiceContext] = None,
        vector_store: Optional[PineconeVectorStore] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        pinecone_kwargs = pinecone_kwargs or {}

        if vector_store is None:
            vector_store = PineconeVectorStore(
                pinecone_index=pinecone_index,
                index_name=index_name,
                environment=environment,
                metadata_filters=metadata_filters,
                pinecone_kwargs=pinecone_kwargs,
                insert_kwargs=insert_kwargs,
                query_kwargs=query_kwargs,
                delete_kwargs=delete_kwargs,
            )
        assert vector_store is not None

        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            service_context=service_context,
            vector_store=vector_store,
            **kwargs,
        )


class GPTWeaviateIndex(GPTVectorStoreIndex):
    """GPT Weaviate Index.

    The GPTWeaviateIndex is a data structure where nodes are keyed by
    embeddings, and those embeddings are stored within a Weaviate index.
    During index construction, the document texts are chunked up,
    converted to nodes with text; they are then encoded in
    document embeddings stored within Weaviate.

    During query time, the index uses Weaviate to query for the top
    k most similar nodes, and synthesizes an answer from the
    retrieved nodes.

    Args:
        service_context (ServiceContext): Service context container (contains
            components like LLMPredictor, PromptHelper, etc.).
    """

    index_struct_cls: Type[IndexDict] = WeaviateIndexDict

    def __init__(
        self,
        nodes: Optional[Sequence[Node]] = None,
        service_context: Optional[ServiceContext] = None,
        weaviate_client: Optional[Any] = None,
        class_prefix: Optional[str] = None,
        index_struct: Optional[IndexDict] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        if weaviate_client is None:
            raise ValueError("weaviate_client is required.")
        vector_store = WeaviateVectorStore(
            weaviate_client=weaviate_client, class_prefix=class_prefix
        )

        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            service_context=service_context,
            vector_store=vector_store,
            **kwargs,
        )


class GPTQdrantIndex(GPTVectorStoreIndex):
    """GPT Qdrant Index.

    The GPTQdrantIndex is a data structure where nodes are keyed by
    embeddings, and those embeddings are stored within a Qdrant collection.
    During index construction, the document texts are chunked up,
    converted to nodes with text; they are then encoded in
    document embeddings stored within Qdrant.

    During query time, the index uses Qdrant to query for the top
    k most similar nodes, and synthesizes an answer from the
    retrieved nodes.

    Args:
        service_context (ServiceContext): Service context container (contains
            components like LLMPredictor, PromptHelper, etc.).
        client (Optional[Any]): QdrantClient instance from `qdrant-client` package
        collection_name: (Optional[str]): name of the Qdrant collection
    """

    index_struct_cls: Type[IndexDict] = QdrantIndexDict

    def __init__(
        self,
        nodes: Optional[Sequence[Node]] = None,
        service_context: Optional[ServiceContext] = None,
        client: Optional[Any] = None,
        collection_name: Optional[str] = None,
        index_struct: Optional[IndexDict] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        if client is None:
            raise ValueError("client is required.")
        if collection_name is None:
            raise ValueError("collection_name is required.")
        vector_store = QdrantVectorStore(client=client, collection_name=collection_name)

        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            service_context=service_context,
            vector_store=vector_store,
            **kwargs,
        )


class GPTChromaIndex(GPTVectorStoreIndex):
    """GPT Chroma Index.

    The GPTChromaIndex is a data structure where nodes are keyed by
    embeddings, and those embeddings are stored within a Chroma collection.
    During index construction, the document texts are chunked up,
    converted to nodes with text; they are then encoded in
    document embeddings stored within Chroma.

    During query time, the index uses Chroma to query for the top
    k most similar nodes, and synthesizes an answer from the
    retrieved nodes.

    Args:
        service_context (ServiceContext): Service context container (contains
            components like LLMPredictor, PromptHelper, etc.).
        chroma_collection (Optional[Any]): Collection instance from `chromadb` package.

    """

    index_struct_cls: Type[IndexDict] = ChromaIndexDict

    def __init__(
        self,
        nodes: Optional[Sequence[Node]] = None,
        index_struct: Optional[IndexDict] = None,
        service_context: Optional[ServiceContext] = None,
        chroma_collection: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        if chroma_collection is None:
            raise ValueError("chroma_collection is required.")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            service_context=service_context,
            vector_store=vector_store,
            **kwargs,
        )


class GPTOpensearchIndex(GPTVectorStoreIndex):
    """GPT Opensearch Index.

    The GPTOpensearchIndex is a data structure where nodes are keyed by
    embeddings, and those embeddings are stored in a document that is indexed
    with its embedding as well as its textual data (text field is defined in
    the OpensearchVectorClient).
    During index construction, the document texts are chunked up,
    converted to nodes with text; each node's embedding is computed, and then
    the node's text, along with the embedding, is converted into JSON document that
    is indexed in Opensearch. The embedding data is put into a field with type
    "knn_vector" and the text is put into a standard Opensearch text field.

    During query time, the index performs approximate KNN search using the
    "knn_vector" field that the embeddings were mapped to.

    Args:
        client (Optional[OpensearchVectorClient]): The client which encapsulates
            logic for using Opensearch as a vector store (that is, it holds stuff
            like endpoint, index_name and performs operations like initializing the
            index and adding new doc/embeddings to said index).
        service_context (ServiceContext): Service context container (contains
            components like LLMPredictor, PromptHelper, etc.).
    """

    index_struct_cls: Type[IndexDict] = OpensearchIndexDict

    def __init__(
        self,
        nodes: Optional[Sequence[Node]] = None,
        service_context: Optional[ServiceContext] = None,
        client: Optional[OpensearchVectorClient] = None,
        index_struct: Optional[IndexDict] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        if client is None:
            raise ValueError("client is required.")
        vector_store = OpensearchVectorStore(client)
        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            service_context=service_context,
            vector_store=vector_store,
            **kwargs,
        )


class ChatGPTRetrievalPluginIndex(GPTVectorStoreIndex):
    """ChatGPTRetrievalPlugin index.

    This index directly interfaces with any server that hosts
    the ChatGPT Retrieval Plugin interface:
    https://github.com/openai/chatgpt-retrieval-plugin.

    Args:
        client (Optional[OpensearchVectorClient]): The client which encapsulates
            logic for using Opensearch as a vector store (that is, it holds stuff
            like endpoint, index_name and performs operations like initializing the
            index and adding new doc/embeddings to said index).
        service_context (ServiceContext): Service context container (contains
            components like LLMPredictor, PromptHelper, etc.).
    """

    index_struct_cls: Type[IndexDict] = ChatGPTRetrievalPluginIndexDict

    def __init__(
        self,
        nodes: Optional[Sequence[Node]] = None,
        index_struct: Optional[ChatGPTRetrievalPluginIndexDict] = None,
        service_context: Optional[ServiceContext] = None,
        endpoint_url: Optional[str] = None,
        bearer_token: Optional[str] = None,
        retries: Optional[Retry] = None,
        batch_size: int = 100,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        if endpoint_url is None:
            raise ValueError("endpoint_url is required.")
        if bearer_token is None:
            raise ValueError("bearer_token is required.")
        vector_store = ChatGPTRetrievalPluginClient(
            endpoint_url,
            bearer_token,
            retries=retries,
            batch_size=batch_size,
        )
        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            service_context=service_context,
            vector_store=vector_store,
            **kwargs,
        )
