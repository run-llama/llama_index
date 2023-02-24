"""Deprecated vector store indices."""

from typing import Any, Optional, Sequence

from gpt_index.data_structs.data_structs import IndexDict
from gpt_index.embeddings.base import BaseEmbedding
from gpt_index.indices.base import DOCUMENTS_INPUT, BaseGPTIndex
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

    def __init__(
        self,
        documents: Optional[Sequence[DOCUMENTS_INPUT]] = None,
        pinecone_index: Optional[Any] = None,
        index_struct: Optional[IndexDict] = None,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        llm_predictor: Optional[LLMPredictor] = None,
        embed_model: Optional[BaseEmbedding] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        if pinecone_index is None:
            raise ValueError("pinecone_index is required.")
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

        super().__init__(
            documents=documents,
            index_struct=index_struct,
            text_qa_template=text_qa_template,
            llm_predictor=llm_predictor,
            embed_model=embed_model,
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
        text_qa_template (Optional[QuestionAnswerPrompt]): A Question-Answer Prompt
            (see :ref:`Prompt-Templates`).
        embed_model (Optional[BaseEmbedding]): Embedding model to use for
            embedding similarity.
    """

    def __init__(
        self,
        documents: Optional[Sequence[DOCUMENTS_INPUT]] = None,
        weaviate_client: Optional[Any] = None,
        index_struct: Optional[IndexDict] = None,
        text_qa_template: Optional[QuestionAnswerPrompt] = None,
        llm_predictor: Optional[LLMPredictor] = None,
        embed_model: Optional[BaseEmbedding] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        if weaviate_client is None:
            raise ValueError("weaviate_client is required.")
        vector_store = WeaviateVectorStore(weaviate_client=weaviate_client)

        super().__init__(
            documents=documents,
            index_struct=index_struct,
            text_qa_template=text_qa_template,
            llm_predictor=llm_predictor,
            embed_model=embed_model,
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
        text_qa_template (Optional[QuestionAnswerPrompt]): A Question-Answer Prompt
            (see :ref:`Prompt-Templates`).
        embed_model (Optional[BaseEmbedding]): Embedding model to use for
            embedding similarity.
        client (Optional[Any]): QdrantClient instance from `qdrant-client` package
        collection_name: (Optional[str]): name of the Qdrant collection
    """

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
        chroma_collection (Optional[Any]): Collection instance from `chromadb` package

    """

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
