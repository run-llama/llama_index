"""Embedding utils for queries."""

from typing import Callable, Dict, List, Optional, Tuple

from dataclasses import field, dataclass
from gpt_index.data_structs.node_v2 import Node, NodeWithScore
from gpt_index.embeddings.base import similarity as default_similarity_fn
import numpy as np
from gpt_index.vector_stores.types import VectorStoreQueryMode


def get_top_k_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    similarity_fn: Optional[Callable[..., float]] = None,
    similarity_top_k: Optional[int] = None,
    embedding_ids: Optional[List] = None,
    similarity_cutoff: Optional[float] = None,
) -> Tuple[List[float], List]:
    """Get top nodes by similarity to the query."""
    if embedding_ids is None:
        embedding_ids = [i for i in range(len(embeddings))]

    similarity_fn = similarity_fn or default_similarity_fn

    similarities = []
    for emb in embeddings:
        similarity = similarity_fn(query_embedding, emb)
        similarities.append(similarity)

    sorted_tups = sorted(
        zip(similarities, embedding_ids), key=lambda x: x[0], reverse=True
    )

    if similarity_cutoff is not None:
        sorted_tups = [tup for tup in sorted_tups if tup[0] > similarity_cutoff]

    similarity_top_k = similarity_top_k or len(sorted_tups)
    result_tups = sorted_tups[:similarity_top_k]

    result_similarities = [s for s, _ in result_tups]
    result_ids = [n for _, n in result_tups]

    return result_similarities, result_ids


def get_top_k_embeddings_learner(
    query_embedding: List[float],
    embeddings: List[List[float]],
    similarity_top_k: Optional[int] = None,
    embedding_ids: Optional[List] = None,
    query_mode: VectorStoreQueryMode = VectorStoreQueryMode.SVM,
) -> Tuple[List[float], List]:
    """Get top embeddings by fitting a learner against query.

    Inspired by Karpathy's SVM demo:
    https://github.com/karpathy/randomfun/blob/master/knn_vs_svm.ipynb

    Can fit SVM, linear regression, and more.

    """
    try:
        from sklearn import svm, linear_model
    except ImportError:
        raise ImportError("Please install scikit-learn to use this feature.")

    if embedding_ids is None:
        embedding_ids = [i for i in range(len(embeddings))]
    query_embedding_np = np.array(query_embedding)
    embeddings_np = np.array(embeddings)
    # create dataset
    dataset_len = len(embeddings) + 1
    dataset = np.concatenate([query_embedding_np[None, ...], embeddings_np])
    y = np.zeros(dataset_len)
    y[0] = 1

    if query_mode == VectorStoreQueryMode.SVM:
        # train our SVM
        # TODO: make params configurable
        clf = svm.LinearSVC(
            class_weight="balanced", verbose=False, max_iter=10000, tol=1e-6, C=0.1
        )
    elif query_mode == VectorStoreQueryMode.LINEAR_REGRESSION:
        clf = linear_model.LinearRegression()
    elif query_mode == VectorStoreQueryMode.LOGISTIC_REGRESSION:
        clf = linear_model.LogisticRegression(class_weight="balanced")
    else:
        raise ValueError(f"Unknown query mode: {query_mode}")

    clf.fit(dataset, y)  # train

    # infer on whatever data you wish, e.g. the original data
    similarities = clf.decision_function(dataset[1:])
    sorted_ix = np.argsort(-similarities)
    top_sorted_ix = sorted_ix[:similarity_top_k]

    result_similarities = similarities[top_sorted_ix]
    result_ids = [embedding_ids[ix] for ix in top_sorted_ix]

    return result_similarities, result_ids


@dataclass
class SimilarityTracker:
    """Helper class to manage node similarities during lifecycle of a single query."""

    # TODO: smarter way to store this information
    lookup: Dict[str, float] = field(default_factory=dict)

    def _hash(self, node: Node) -> str:
        """Generate a unique key for each node."""
        # TODO: Better way to get unique identifier of a node
        return str(abs(hash(node.get_text())))

    def add(self, node: Node, similarity: float) -> None:
        """Add a node and its similarity score."""
        node_hash = self._hash(node)
        self.lookup[node_hash] = similarity

    def find(self, node: Node) -> Optional[float]:
        """Find a node's similarity score."""
        node_hash = self._hash(node)
        if node_hash not in self.lookup:
            return None
        return self.lookup[node_hash]

    def get_zipped_nodes(self, nodes: List[Node]) -> List[NodeWithScore]:
        """Get a zipped list of nodes and their corresponding scores."""
        similarities = [self.find(node) for node in nodes]
        output = []
        for node, score in zip(nodes, similarities):
            output.append(NodeWithScore(node=node, score=score))
        return output
