import logging
from typing import Callable, List, Optional, cast

from llama_index.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core import BaseRetriever
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.schema import BaseNode, NodeWithScore, QueryBundle
from llama_index.storage.docstore.types import BaseDocumentStore
from llama_index.utils import get_tokenizer

logger = logging.getLogger(__name__)


class BM25Retriever(BaseRetriever):
    def __init__(
        self,
        nodes: List[BaseNode],
        tokenizer: Optional[Callable[[str], List[str]]],
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
    ) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Please install rank_bm25: pip install rank-bm25")

        self._nodes = nodes
        self._tokenizer = tokenizer or (lambda x: x.split(" "))
        self._similarity_top_k = similarity_top_k
        self._corpus = [self._tokenizer(node.get_content()) for node in self._nodes]

        self.bm25 = BM25Okapi(self._corpus)

    @classmethod
    def from_defaults(
        cls,
        index: Optional[VectorStoreIndex] = None,
        nodes: Optional[List[BaseNode]] = None,
        docstore: Optional[BaseDocumentStore] = None,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
    ) -> "BM25Retriever":
        # ensure only one of index, nodes, or docstore is passed
        if sum(bool(val) for val in [index, nodes, docstore]) != 1:
            raise ValueError("Please pass exactly one of index, nodes, or docstore.")

        if index is not None:
            docstore = index.docstore

        if docstore is not None:
            nodes = cast(List[BaseNode], list(docstore.docs.values()))

        assert (
            nodes is not None
        ), "Please pass exactly one of index, nodes, or docstore."

        tokenizer = tokenizer or get_tokenizer()
        return cls(
            nodes=nodes,
            tokenizer=tokenizer,
            similarity_top_k=similarity_top_k,
        )

    def _get_scored_nodes(self, query: str) -> List[NodeWithScore]:
        tokenized_query = self._tokenizer(query)
        doc_scores = self.bm25.get_scores(tokenized_query)

        nodes: List[NodeWithScore] = []
        for i, node in enumerate(self._nodes):
            nodes.append(NodeWithScore(node=node, score=doc_scores[i]))

        return nodes

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        if query_bundle.custom_embedding_strs or query_bundle.embedding:
            logger.warning("BM25Retriever does not support embeddings, skipping...")

        scored_nodes = self._get_scored_nodes(query_bundle.query_str)

        # Sort and get top_k nodes, score range => 0..1, closer to 1 means more relevant
        nodes = sorted(scored_nodes, key=lambda x: x.score or 0.0, reverse=True)
        return nodes[: self._similarity_top_k]
