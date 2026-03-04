"""DeepMemory Retrieval Pack."""

from typing import Any, Dict, List, Optional

from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import TextNode
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.deeplake import DeepLakeVectorStore


class DeepMemoryRetrieverPack(BaseLlamaPack):
    """DeepMemory retriever pack."""

    def __init__(
        self,
        dataset_path: str = "llama_index",
        token: Optional[str] = None,
        read_only: Optional[bool] = False,
        overwrite: bool = False,
        verbose: bool = True,
        nodes: Optional[List[TextNode]] = None,
        top_k: int = 4,
        **kwargs: Any,
    ):
        self._vector_store = DeepLakeVectorStore(
            dataset_path=dataset_path,
            token=token,
            read_only=read_only,
            overwrite=overwrite,
            verbose=verbose,
        )

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

        self.retriever = self._index.as_retriever(
            similarity_top_k=top_k, vector_store_kwargs={"deep_memory": True}
        )
        self.query_engine = RetrieverQueryEngine.from_args(retriever=self.retriever)

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
