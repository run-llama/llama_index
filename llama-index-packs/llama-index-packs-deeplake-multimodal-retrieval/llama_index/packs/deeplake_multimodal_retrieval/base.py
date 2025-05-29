"""DeepLake multimodal Retrieval Pack."""


from typing import Any, Dict, List, Optional

from llama_index.core.indices.multi_modal import MultiModalVectorStoreIndex
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.query_engine import SimpleMultiModalQueryEngine
from llama_index.core.schema import BaseNode
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.deeplake import DeepLakeVectorStore


class DeepLakeMultimodalRetrieverPack(BaseLlamaPack):
    """DeepLake Multimodal retriever pack."""

    def __init__(
        self,
        dataset_path: str = "llama_index",
        token: Optional[str] = None,
        read_only: Optional[bool] = False,
        overwrite: bool = False,
        verbose: bool = True,
        nodes: Optional[List[BaseNode]] = None,
        top_k: int = 4,
        **kwargs: Any,
    ):
        # text vector store
        self._text_vectorstore = DeepLakeVectorStore(
            dataset_path=dataset_path + "_text",
            token=token,
            read_only=read_only,
            overwrite=overwrite,
            verbose=verbose,
        )

        # image vector store
        self._image_vectorstore = DeepLakeVectorStore(
            dataset_path=dataset_path + "_image",
            token=token,
            read_only=read_only,
            overwrite=overwrite,
            verbose=verbose,
        )

        if nodes is not None:
            self._storage_context = StorageContext.from_defaults(
                vector_store=self._text_vectorstore
            )
            self._index = MultiModalVectorStoreIndex(
                nodes,
                storage_context=self._storage_context,
                image_vector_store=self._image_vectorstore,
            )
        else:
            self._storage_context = StorageContext.from_defaults(
                vector_store=self._text_vectorstore
            )
            self._index = MultiModalVectorStoreIndex.from_vector_store(
                self._text_vectorstore,
                image_vector_store=self._image_vectorstore,
            )
        self.retriever = self._index.as_retriever(
            similarity_top_k=top_k, vector_store_kwargs={"deep_memory": True}
        )
        self.query_engine = SimpleMultiModalQueryEngine(self.retriever)

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {
            "text_vectorstore": self._text_vectorstore,
            "image_vectorstore": self._image_vectorstore,
            "storage_context": self._storage_context,
            "index": self._index,
            "retriever": self.retriever,
            "query_engine": self.query_engine,
        }

    def retrieve(self, query_str: str) -> Any:
        """Retrieve."""
        return self.query_engine.retrieve(query_str)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        return self.query_engine.query(*args, **kwargs)
