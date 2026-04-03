"""Hybrid Fusion Retriever Pack."""

import os
from typing import Any, Dict, List

from llama_index.core import Settings
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import Document, TextNode
from llama_index.retrievers.bm25 import BM25Retriever


class HybridFusionRetrieverPack(BaseLlamaPack):
    """
    Hybrid fusion retriever pack.

    Ensembles vector and bm25 retrievers using fusion.

    """

    def __init__(
        self,
        nodes: List[TextNode] = None,
        chunk_size: int = 256,
        mode: str = "reciprocal_rerank",
        vector_similarity_top_k: int = 2,
        bm25_similarity_top_k: int = 2,
        fusion_similarity_top_k: int = 2,
        num_queries: int = 4,
        documents: List[Document] = None,
        cache_dir: str = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        Settings.chunk_size = chunk_size
        if cache_dir is not None and os.path.exists(cache_dir):
            # Load from cache
            from llama_index import StorageContext, load_index_from_storage

            # rebuild storage context
            storage_context = StorageContext.from_defaults(persist_dir=cache_dir)
            # load index
            index = load_index_from_storage(storage_context)
        elif documents is not None:
            index = VectorStoreIndex.from_documents(documents=documents)
        else:
            index = VectorStoreIndex(nodes)

        if cache_dir is not None and not os.path.exists(cache_dir):
            index.storage_context.persist(persist_dir=cache_dir)

        self.vector_retriever = index.as_retriever(
            similarity_top_k=vector_similarity_top_k
        )

        self.bm25_retriever = BM25Retriever.from_defaults(
            docstore=index.docstore, similarity_top_k=bm25_similarity_top_k
        )
        self.fusion_retriever = QueryFusionRetriever(
            [self.vector_retriever, self.bm25_retriever],
            similarity_top_k=fusion_similarity_top_k,
            num_queries=num_queries,  # set this to 1 to disable query generation
            mode=mode,
            use_async=True,
            verbose=True,
            # query_gen_prompt="...",  # we could override the query generation prompt here
        )

        self.query_engine = RetrieverQueryEngine.from_args(self.fusion_retriever)

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {
            "vector_retriever": self.vector_retriever,
            "bm25_retriever": self.bm25_retriever,
            "fusion_retriever": self.fusion_retriever,
            "query_engine": self.query_engine,
        }

    def retrieve(self, query_str: str) -> Any:
        """Retrieve."""
        return self.fusion_retriever.retrieve(query_str)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        return self.query_engine.query(*args, **kwargs)
