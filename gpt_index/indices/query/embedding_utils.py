"""Embedding utils for queries."""

from typing import Callable, List, Optional, Tuple

from gpt_index.embeddings.base import similarity as default_similarity_fn


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
