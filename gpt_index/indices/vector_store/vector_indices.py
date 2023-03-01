"""Deprecated vector store indices."""

from typing import Any, Dict, Optional, Sequence, Type, cast

from gpt_index.data_structs.data_structs import (
    ChromaIndexDict,
    FaissIndexDict,
    IndexDict,
    OpensearchIndexDict,
    PineconeIndexDict,
    QdrantIndexDict,
    SimpleIndexDict,
    WeaviateIndexDict,
)
from gpt_index.embeddings.base import BaseEmbedding
from gpt_index.indices.base import DOCUMENTS_INPUT, BaseGPTIndex
from gpt_index.indices.query.base import BaseGPTIndexQuery
from gpt_index.indices.query.schema import QueryMode
from gpt_index.indices.query.vector_store.queries import (
    GPTChromaIndexQuery,
    GPTFaissIndexQuery,
    GPTOpensearchIndexQuery,
    GPTPineconeIndexQuery,
    GPTQdrantIndexQuery,
    GPTSimpleVectorIndexQuery,
    GPTWeaviateIndexQuery,
)
from gpt_index.indices.vector_store.base import GPTVectorStoreIndex
from gpt_index.langchain_helpers.chain_wrapper import LLMPredictor
from gpt_index.prompts.prompts import QuestionAnswerPrompt
from gpt_index.vector_stores import (
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
        text_qa_template (Optional[QuestionAnswerPrompt]): A Question-Answer Prompt
            (see :ref:`Prompt-Templates`).
        embed_model (Optional[BaseEmbedding]): Embedding model to use for
            embedding similarity.

    """

    index_struct_cls: Type[IndexDict] = SimpleIndexDict

    def __init__(
        self,
        documents: Optional[Sequence[DOCUMENTS_INPUT]] = None,
        index_struct: Optional[IndexDict] = None,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        llm_predictor: Optional[LLMPredictor] = None,
        embed_model: Optional[BaseEmbedding] = None,
        simple_vector_store_data_dict: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        vector_store = SimpleVectorStore(
            simple_vector_store_data_dict=simple_vector_store_data_dict
        )

        super().__init__(
            documents=documents,
            index_struct=index_struct,
            text_qa_template=text_qa_template,
            llm_predictor=llm_predictor,
            embed_model=embed_model,
            vector_store=vector_store,
            **kwargs,
        )

        # TODO: Temporary hack to also store embeddings in index_struct
        embedding_dict = vector_store._data.embedding_dict
        self._index_struct.embeddings_dict = embedding_dict
        # update docstore with current struct
        self._docstore.add_documents([self.index_struct], allow_update=True)

    @classmethod
    def get_query_map(self) -> Dict[str, Type[BaseGPTIndexQuery]]:
        """Get query map."""
        return {
            QueryMode.DEFAULT: GPTSimpleVectorIndexQuery,
            QueryMode.EMBEDDING: GPTSimpleVectorIndexQuery,
        }

    def _preprocess_query(self, mode: QueryMode, query_kwargs: Any) -> None:
        """Preprocess query."""
        super()._preprocess_query(mode, query_kwargs)
        del query_kwargs["vector_store"]
        vector_store = cast(SimpleVectorStore, self._vector_store)
        query_kwargs["simple_vector_store_data_dict"] = vector_store._data


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
        text_qa_template (Optional[QuestionAnswerPrompt]): A Question-Answer Prompt
            (see :ref:`Prompt-Templates`).
        faiss_index (faiss.Index): A Faiss Index object (required). Note: the index
            will be reset during index construction.
        embed_model (Optional[BaseEmbedding]): Embedding model to use for
            embedding similarity.
    """

    index_struct_cls: Type[IndexDict] = FaissIndexDict

    def __init__(
        self,
        documents: Optional[Sequence[DOCUMENTS_INPUT]] = None,
        faiss_index: Optional[Any] = None,
        index_struct: Optional[IndexDict] = None,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        llm_predictor: Optional[LLMPredictor] = None,
        embed_model: Optional[BaseEmbedding] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        if faiss_index is None:
            raise ValueError("faiss_index is required.")
        vector_store = FaissVectorStore(faiss_index)

        super().__init__(
            documents=documents,
            index_struct=index_struct,
            text_qa_template=text_qa_template,
            llm_predictor=llm_predictor,
            embed_model=embed_model,
            vector_store=vector_store,
            **kwargs,
        )

    @classmethod
    def get_query_map(self) -> Dict[str, Type[BaseGPTIndexQuery]]:
        """Get query map."""
        return {
            QueryMode.DEFAULT: GPTFaissIndexQuery,
            QueryMode.EMBEDDING: GPTFaissIndexQuery,
        }

    def _preprocess_query(self, mode: QueryMode, query_kwargs: Any) -> None:
        """Preprocess query."""
        super()._preprocess_query(mode, query_kwargs)
        del query_kwargs["vector_store"]
        vector_store = cast(FaissVectorStore, self._vector_store)
        query_kwargs["faiss_index"] = vector_store._faiss_index

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
            faiss_index_save_path (Optional[str]): The save_path of the
                Faiss index file. If not specified, the Faiss index
                will not be saved to disk.
        """
        super().save_to_disk(save_path, **save_kwargs)

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
        text_qa_template (Optional[QuestionAnswerPrompt]): A Question-Answer Prompt
            (see :ref:`Prompt-Templates`).
        embed_model (Optional[BaseEmbedding]): Embedding model to use for
            embedding similarity.
        chunk_size_limit (int): Maximum number of tokens per chunk. NOTE:
            in Pinecone the default is 2048 due to metadata size restrictions.
    """

    index_struct_cls: Type[IndexDict] = PineconeIndexDict

    def __init__(
        self,
        documents: Optional[Sequence[DOCUMENTS_INPUT]] = None,
        pinecone_index: Optional[Any] = None,
        pinecone_kwargs: Optional[Dict] = None,
        index_struct: Optional[IndexDict] = None,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        llm_predictor: Optional[LLMPredictor] = None,
        embed_model: Optional[BaseEmbedding] = None,
        chunk_size_limit: int = 2048,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        if pinecone_index is None:
            raise ValueError("pinecone_index is required.")
        if pinecone_kwargs is None:
            pinecone_kwargs = {}
        vector_store = kwargs.pop(
            "vector_store",
            PineconeVectorStore(
                pinecone_index=pinecone_index, pinecone_kwargs=pinecone_kwargs
            ),
        )

        super().__init__(
            documents=documents,
            index_struct=index_struct,
            text_qa_template=text_qa_template,
            llm_predictor=llm_predictor,
            embed_model=embed_model,
            vector_store=vector_store,
            chunk_size_limit=chunk_size_limit,
            **kwargs,
        )

    @classmethod
    def get_query_map(self) -> Dict[str, Type[BaseGPTIndexQuery]]:
        """Get query map."""
        return {
            QueryMode.DEFAULT: GPTPineconeIndexQuery,
            QueryMode.EMBEDDING: GPTPineconeIndexQuery,
        }

    def _preprocess_query(self, mode: QueryMode, query_kwargs: Any) -> None:
        """Preprocess query."""
        super()._preprocess_query(mode, query_kwargs)
        del query_kwargs["vector_store"]
        vector_store = cast(PineconeVectorStore, self._vector_store)
        query_kwargs["pinecone_index"] = vector_store._pinecone_index
        query_kwargs["pinecone_kwargs"] = vector_store._pinecone_kwargs


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
        text_qa_template (Optional[QuestionAnswerPrompt]): A Question-Answer Prompt
            (see :ref:`Prompt-Templates`).
        embed_model (Optional[BaseEmbedding]): Embedding model to use for
            embedding similarity.
    """

    index_struct_cls: Type[IndexDict] = WeaviateIndexDict

    def __init__(
        self,
        documents: Optional[Sequence[DOCUMENTS_INPUT]] = None,
        weaviate_client: Optional[Any] = None,
        class_prefix: Optional[str] = None,
        index_struct: Optional[IndexDict] = None,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        llm_predictor: Optional[LLMPredictor] = None,
        embed_model: Optional[BaseEmbedding] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        if weaviate_client is None:
            raise ValueError("weaviate_client is required.")
        vector_store = WeaviateVectorStore(
            weaviate_client=weaviate_client, class_prefix=class_prefix
        )

        super().__init__(
            documents=documents,
            index_struct=index_struct,
            text_qa_template=text_qa_template,
            llm_predictor=llm_predictor,
            embed_model=embed_model,
            vector_store=vector_store,
            **kwargs,
        )

    @classmethod
    def get_query_map(self) -> Dict[str, Type[BaseGPTIndexQuery]]:
        """Get query map."""
        return {
            QueryMode.DEFAULT: GPTWeaviateIndexQuery,
            QueryMode.EMBEDDING: GPTWeaviateIndexQuery,
        }

    def _preprocess_query(self, mode: QueryMode, query_kwargs: Any) -> None:
        """Preprocess query."""
        super()._preprocess_query(mode, query_kwargs)
        del query_kwargs["vector_store"]
        vector_store = cast(WeaviateVectorStore, self._vector_store)
        query_kwargs["weaviate_client"] = vector_store._client
        query_kwargs["class_prefix"] = vector_store._class_prefix


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
        text_qa_template (Optional[QuestionAnswerPrompt]): A Question-Answer Prompt
            (see :ref:`Prompt-Templates`).
        embed_model (Optional[BaseEmbedding]): Embedding model to use for
            embedding similarity.
        client (Optional[Any]): QdrantClient instance from `qdrant-client` package
        collection_name: (Optional[str]): name of the Qdrant collection
    """

    index_struct_cls: Type[IndexDict] = QdrantIndexDict

    def __init__(
        self,
        documents: Optional[Sequence[DOCUMENTS_INPUT]] = None,
        client: Optional[Any] = None,
        collection_name: Optional[str] = None,
        index_struct: Optional[IndexDict] = None,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        llm_predictor: Optional[LLMPredictor] = None,
        embed_model: Optional[BaseEmbedding] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        if client is None:
            raise ValueError("client is required.")
        if collection_name is None:
            raise ValueError("collection_name is required.")
        vector_store = QdrantVectorStore(client=client, collection_name=collection_name)

        super().__init__(
            documents=documents,
            index_struct=index_struct,
            text_qa_template=text_qa_template,
            llm_predictor=llm_predictor,
            embed_model=embed_model,
            vector_store=vector_store,
            **kwargs,
        )

    @classmethod
    def get_query_map(self) -> Dict[str, Type[BaseGPTIndexQuery]]:
        """Get query map."""
        return {
            QueryMode.DEFAULT: GPTQdrantIndexQuery,
            QueryMode.EMBEDDING: GPTQdrantIndexQuery,
        }

    def _preprocess_query(self, mode: QueryMode, query_kwargs: Any) -> None:
        """Preprocess query."""
        super()._preprocess_query(mode, query_kwargs)
        del query_kwargs["vector_store"]
        vector_store = cast(QdrantVectorStore, self._vector_store)
        query_kwargs["client"] = vector_store._client
        query_kwargs["collection_name"] = vector_store._collection_name


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
        text_qa_template (Optional[QuestionAnswerPrompt]): A Question-Answer Prompt
            (see :ref:`Prompt-Templates`).
        embed_model (Optional[BaseEmbedding]): Embedding model to use for
            embedding similarity.
        chroma_collection (Optional[Any]): Collection instance from `chromadb` package.

    """

    index_struct_cls: Type[IndexDict] = ChromaIndexDict

    def __init__(
        self,
        documents: Optional[Sequence[DOCUMENTS_INPUT]] = None,
        chroma_collection: Optional[Any] = None,
        index_struct: Optional[IndexDict] = None,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        llm_predictor: Optional[LLMPredictor] = None,
        embed_model: Optional[BaseEmbedding] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        if chroma_collection is None:
            raise ValueError("chroma_collection is required.")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        super().__init__(
            documents=documents,
            index_struct=index_struct,
            text_qa_template=text_qa_template,
            llm_predictor=llm_predictor,
            embed_model=embed_model,
            vector_store=vector_store,
            **kwargs,
        )

    @classmethod
    def get_query_map(self) -> Dict[str, Type[BaseGPTIndexQuery]]:
        """Get query map."""
        return {
            QueryMode.DEFAULT: GPTChromaIndexQuery,
            QueryMode.EMBEDDING: GPTChromaIndexQuery,
        }

    def _preprocess_query(self, mode: QueryMode, query_kwargs: Any) -> None:
        """Preprocess query."""
        super()._preprocess_query(mode, query_kwargs)
        del query_kwargs["vector_store"]
        vector_store = cast(ChromaVectorStore, self._vector_store)
        query_kwargs["chroma_collection"] = vector_store._collection


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
        text_qa_template (Optional[QuestionAnswerPrompt]): A Question-Answer Prompt
            (see :ref:`Prompt-Templates`).
        client (Optional[OpensearchVectorClient]): The client which encapsulates
            logic for using Opensearch as a vector store (that is, it holds stuff
            like endpoint, index_name and performs operations like initializing the
            index and adding new doc/embeddings to said index).
        embed_model (Optional[BaseEmbedding]): Embedding model to use for
            embedding similarity.
    """

    index_struct_cls: Type[IndexDict] = OpensearchIndexDict

    def __init__(
        self,
        documents: Optional[Sequence[DOCUMENTS_INPUT]] = None,
        client: Optional[OpensearchVectorClient] = None,
        index_struct: Optional[IndexDict] = None,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        llm_predictor: Optional[LLMPredictor] = None,
        embed_model: Optional[BaseEmbedding] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        if client is None:
            raise ValueError("client is required.")
        vector_store = OpensearchVectorStore(client)
        super().__init__(
            documents=documents,
            index_struct=index_struct,
            text_qa_template=text_qa_template,
            llm_predictor=llm_predictor,
            embed_model=embed_model,
            vector_store=vector_store,
            **kwargs,
        )

    @classmethod
    def get_query_map(self) -> Dict[str, Type[BaseGPTIndexQuery]]:
        """Get query map."""
        return {
            QueryMode.DEFAULT: GPTOpensearchIndexQuery,
            QueryMode.EMBEDDING: GPTOpensearchIndexQuery,
        }

    def _preprocess_query(self, mode: QueryMode, query_kwargs: Any) -> None:
        """Preprocess query."""
        super()._preprocess_query(mode, query_kwargs)
        del query_kwargs["vector_store"]
        vector_store = cast(OpensearchVectorStore, self._vector_store)
        query_kwargs["client"] = vector_store._client
