import logging
from collections.abc import Callable
from typing import Callable, Optional, cast, List

from llama_index.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.schema import Document, NodeWithScore
from llama_index.storage.docstore.types import BaseDocumentStore
from llama_index.utils import globals_helper

logger = logging.getLogger(__name__)


class BM25Retriever(BaseRetriever):
    def __init__(
        self,
        docstore: BaseDocumentStore,
        tokenizer: Callable[[str], List[str]],
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
    ) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Please install rank_bm25: pip install rank-bm25")

        self._docstore = docstore
        self._tokenizer = tokenizer
        self._similarity_top_k = similarity_top_k
        self._documents = cast(
            list[Document], [doc for doc in self._docstore.docs.values()]
        )
        self._corpus = [self._tokenizer(doc.text) for doc in self._documents]

        self.bm25 = BM25Okapi(self._corpus)

    @classmethod
    def from_defaults(
        cls,
        index: VectorStoreIndex,
        tokenizer: Optional[Callable[[str], list[str]]] = None,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
    ) -> "BM25Retriever":
        tokenizer = tokenizer or globals_helper.tokenizer
        return cls(
            index.docstore,
            tokenizer,
            similarity_top_k=similarity_top_k,
        )

    def _get_scored_nodes(self, query: str) -> list[NodeWithScore]:
        tokenized_query = self._tokenizer(query)
        doc_scores = self.bm25.get_scores(tokenized_query)

        nodes: list[NodeWithScore] = []
        for i, doc in enumerate(self._documents):
            nodes.append(NodeWithScore(node=doc, score=doc_scores[i]))

        return nodes

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        if query_bundle.custom_embedding_strs or query_bundle.embedding:
            logger.warning("BM25Retriever does not support embeddings, skipping...")

        scored_nodes = self._get_scored_nodes(query_bundle.query_str)

        # Sort and get top_k nodes, score range => 0..1, closer to 1 means more relevant
        nodes = sorted(scored_nodes, key=lambda x: x.score or 0.0, reverse=True)
        top_k_nodes = nodes[: self._similarity_top_k]

        return top_k_nodes
