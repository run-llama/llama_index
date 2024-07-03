import logging
from typing import Callable, List, Optional, cast

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.indices.keyword_table.utils import simple_extract_keywords
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.schema import BaseNode, IndexNode, NodeWithScore, QueryBundle
from llama_index.core.storage.docstore.types import BaseDocumentStore
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def tokenize_remove_stopwords(text: str) -> List[str]:
    # lowercase and stem words
    text = text.lower()
    stemmer = PorterStemmer()
    words = list(simple_extract_keywords(text))
    return [stemmer.stem(word) for word in words]


class BM25Retriever(BaseRetriever):
    def __init__(
        self,
        nodes: List[BaseNode],
        tokenizer: Optional[Callable[[str], List[str]]],
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        callback_manager: Optional[CallbackManager] = None,
        objects: Optional[List[IndexNode]] = None,
        object_map: Optional[dict] = None,
        verbose: bool = False,
    ) -> None:
        self._nodes = nodes
        self._tokenizer = tokenizer or tokenize_remove_stopwords
        self._similarity_top_k = similarity_top_k
        self._corpus = [self._tokenizer(node.get_content()) for node in self._nodes]
        self.bm25 = BM25Okapi(self._corpus)
        super().__init__(
            callback_manager=callback_manager,
            object_map=object_map,
            objects=objects,
            verbose=verbose,
        )

    @classmethod
    def from_defaults(
        cls,
        index: Optional[VectorStoreIndex] = None,
        nodes: Optional[List[BaseNode]] = None,
        docstore: Optional[BaseDocumentStore] = None,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        verbose: bool = False,
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

        tokenizer = tokenizer or tokenize_remove_stopwords
        return cls(
            nodes=nodes,
            tokenizer=tokenizer,
            similarity_top_k=similarity_top_k,
            verbose=verbose,
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        if query_bundle.custom_embedding_strs or query_bundle.embedding:
            logger.warning("BM25Retriever does not support embeddings, skipping...")

        query = query_bundle.query_str
        tokenized_query = self._tokenizer(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_n = scores.argsort()[::-1][: self._similarity_top_k]

        nodes: List[NodeWithScore] = []
        for ix in top_n:
            nodes.append(NodeWithScore(node=self._nodes[ix], score=float(scores[ix])))

        return nodes
