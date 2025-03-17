"""Chroma Auto-retrieval Pack."""

from typing import Any, Dict, List, Optional

from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.indices.vector_store.retrievers import (
    VectorIndexAutoRetriever,
)
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import TextNode
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.vector_stores.types import VectorStoreInfo
from llama_index.vector_stores.chroma import ChromaVectorStore


class ChromaAutoretrievalPack(BaseLlamaPack):
    """Chroma auto-retrieval pack."""

    def __init__(
        self,
        collection_name: str,
        vector_store_info: VectorStoreInfo,
        nodes: Optional[List[TextNode]] = None,
        client: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        import chromadb

        chroma_client = client or chromadb.EphemeralClient()
        chroma_collection = chroma_client.get_or_create_collection(collection_name)

        self._vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        if nodes is not None:
            self._storage_context = StorageContext.from_defaults(
                vector_store=self._vector_store
            )
            self._index = VectorStoreIndex(
                nodes, storage_context=self._storage_context, **kwargs
            )
        else:
            self._index = VectorStoreIndex.from_vector_store(
                self._vector_store, **kwargs
            )
            self._storage_context = self._index.storage_context

        self.retriever = VectorIndexAutoRetriever(
            self._index, vector_store_info=vector_store_info
        )
        self.query_engine = RetrieverQueryEngine(self.retriever)

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {
            "vector_store": self._vector_store,
            "storage_context": self._storage_context,
            "index": self._index,
            "retriever": self.retriever,
            "query_engine": self.query_engine,
        }

    def retrieve(self, query_str: str) -> Any:
        """Retrieve."""
        return self.retriever.retrieve(query_str)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        return self.query_engine.query(*args, **kwargs)
