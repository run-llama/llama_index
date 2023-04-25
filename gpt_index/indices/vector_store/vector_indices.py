"""Deprecated vector store indices."""

from typing import Any, Callable, Dict, Optional, Sequence, Type

from requests.adapters import Retry

from gpt_index.data_structs.data_structs_v2 import (
    ChatGPTRetrievalPluginIndexDict,
    ChromaIndexDict,
    FaissIndexDict,
    IndexDict,
    MilvusIndexDict,
    MyScaleIndexDict,
    OpensearchIndexDict,
    PineconeIndexDict,
    QdrantIndexDict,
    SimpleIndexDict,
    WeaviateIndexDict,
    DeepLakeIndexDict,
)
from gpt_index.data_structs.node_v2 import Node
from gpt_index.indices.base import BaseGPTIndex
from gpt_index.indices.service_context import ServiceContext
from gpt_index.indices.vector_store.base import GPTVectorStoreIndex
from gpt_index.vector_stores import (
    ChatGPTRetrievalPluginClient,
    ChromaVectorStore,
    DeepLakeVectorStore,
    FaissVectorStore,
    MilvusVectorStore,
    MyScaleVectorStore,
    PineconeVectorStore,
    QdrantVectorStore,
    SimpleVectorStore,
    WeaviateVectorStore,
)
from gpt_index.vector_stores.opensearch import (
    OpensearchVectorClient,
    OpensearchVectorStore,
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
        vector_store: Optional[FaissVectorStore] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        if vector_store is None:
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
