"""Test embedding utility functions."""

import numpy as np
from llama_index.core.indices.query.embedding_utils import (
    get_top_k_embeddings,
    get_top_k_mmr_embeddings,
)


def test_get_top_k_mmr_embeddings() -> None:
    """Test Maximum Marginal Relevance."""
    # Results score should follow from the mmr algorithm
    query_embedding = [5.0, 0.0, 0.0]
    embeddings = [[4.0, 3.0, 0.0], [3.0, 4.0, 0.0], [-4.0, 3.0, 0.0]]
    result_similarities, result_ids = get_top_k_mmr_embeddings(
        query_embedding, embeddings, mmr_threshold=0.8
    )

    assert np.isclose(0.8 * 4 / 5, result_similarities[0], atol=0.00001)
    assert np.isclose(
        0.8 * 3 / 5 - (1 - 0.8) * (3 * 4 / 25 + 3 * 4 / 25),
        result_similarities[1],
        atol=0.00001,
    )
    assert np.isclose(
        0.8 * -4 / 5 - (1 - 0.8) * (3 * -4 / 25 + 4 * 3 / 25),
        result_similarities[2],
        atol=0.00001,
    )
    assert result_ids == [0, 1, 2]

    # Tests that if the first embedding vector is close to the second,
    # it will return the third
    query_embedding = [1.0, 0.0, 1.0]
    embeddings = [[1.0, 0.0, 0.9], [1.0, 0.0, 0.8], [0.7, 0.0, 1.0]]

    _, result_ids = get_top_k_mmr_embeddings(
        query_embedding, embeddings, mmr_threshold=0.5
    )
    assert result_ids == [0, 2, 1]

    # Tests that embedding ids map properly to results
    _, result_ids = get_top_k_mmr_embeddings(
        query_embedding, embeddings, embedding_ids=["A", "B", "C"], mmr_threshold=0.5
    )
    assert result_ids == ["A", "C", "B"]
    # Test that it will go back to the original order under a high threshold
    _, result_ids = get_top_k_mmr_embeddings(
        query_embedding, embeddings, mmr_threshold=1
    )
    assert result_ids == [0, 1, 2]

    # Test similarity_top_k works
    _, result_ids = get_top_k_mmr_embeddings(
        query_embedding, embeddings, mmr_threshold=1, similarity_top_k=2
    )
    assert result_ids == [0, 1]

    # Test the results for get_top_k_embeddings and get_top_k_mmr_embeddings are the
    # same for threshold = 1
    query_embedding = [10, 23, 90, 78]
    embeddings = [[1, 23, 89, 68], [1, 74, 144, 23], [0.23, 0.0, 1.0, 9]]
    result_similarities_no_mmr, result_ids_no_mmr = get_top_k_embeddings(
        query_embedding, embeddings
    )
    result_similarities, result_ids = get_top_k_mmr_embeddings(
        query_embedding, embeddings, mmr_threshold=1
    )

    for result_no_mmr, result_with_mmr in zip(
        result_similarities_no_mmr, result_similarities
    ):
        assert np.isclose(result_no_mmr, result_with_mmr, atol=0.00001)


def test_get_top_k_mmr_embeddings_threshold_zero() -> None:
    """
    An explicit mmr_threshold of 0 must request maximum diversity.

    The docstring promises that "a mmr_threshold of 0 will strongly avoid
    similarity to previous results". Because 0 is falsy, ``mmr_threshold or 0.5``
    silently discarded the value and fell back to 0.5, so the parameter was
    ignored whenever a caller explicitly asked for full diversity.
    """
    query_embedding = [1.0, 0.0]
    # index 0: relevant to the query
    # index 1: a near-duplicate of index 0 (redundant)
    # index 2: orthogonal to the query (maximally diverse)
    embeddings = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

    # With full diversity (threshold=0) the second pick must be the orthogonal
    # node, not the duplicate of the first pick.
    _, result_ids = get_top_k_mmr_embeddings(
        query_embedding, embeddings, similarity_top_k=2, mmr_threshold=0.0
    )
    assert result_ids == [0, 2]

    # A threshold of 0 must behave differently from the 0.5 default, which keeps
    # the redundant high-similarity node.
    _, default_ids = get_top_k_mmr_embeddings(
        query_embedding, embeddings, similarity_top_k=2, mmr_threshold=0.5
    )
    assert default_ids == [0, 1]
    assert result_ids != default_ids

    # Leaving mmr_threshold unset still applies the 0.5 default.
    _, unset_ids = get_top_k_mmr_embeddings(
        query_embedding, embeddings, similarity_top_k=2
    )
    assert unset_ids == [0, 1]


def test_get_top_k_mmr_embeddings_similarity_cutoff() -> None:
    """
    A similarity_cutoff must drop low-similarity candidates before MMR runs.

    ``get_top_k_embeddings`` keeps a candidate only when its similarity to the
    query is strictly greater than the cutoff. ``get_top_k_mmr_embeddings``
    accepted the same ``similarity_cutoff`` argument but never applied it, so a
    caller that asked for a cutoff silently received every candidate back.
    """
    query_embedding = [1.0, 0.0]
    # index 0: highly relevant, index 1: mildly relevant, index 2: orthogonal
    embeddings = [[1.0, 0.0], [0.5, 0.0], [0.0, 1.0]]

    # Without a cutoff every candidate is returned.
    _, all_ids = get_top_k_mmr_embeddings(query_embedding, embeddings, mmr_threshold=1)
    assert all_ids == [0, 1, 2]

    # A cutoff keeps only the candidates above it.
    _, cutoff_ids = get_top_k_mmr_embeddings(
        query_embedding, embeddings, mmr_threshold=1, similarity_cutoff=0.4
    )
    assert cutoff_ids == [0, 1]

    # The cutoff must agree with the non-MMR helper on the same input.
    _, expected_ids = get_top_k_embeddings(
        query_embedding, embeddings, similarity_cutoff=0.4
    )
    assert cutoff_ids == expected_ids

    # A cutoff that no candidate clears leaves nothing to select from.
    similarities, ids = get_top_k_mmr_embeddings(
        query_embedding, embeddings, mmr_threshold=1, similarity_cutoff=1.0
    )
    assert ids == []
    assert similarities == []


def test_get_top_k_mmr_embeddings_cutoff_respects_top_k() -> None:
    """Cutoff filtering must not overrun the pool of surviving candidates."""

    def dot(q, e):
        return float(np.dot(q, e))

    query_embedding = [1.0, 0.0]
    embeddings = [[1.0, 0.0], [0.9, 0.0], [0.0, 1.0], [0.0, 1.0]]

    cutoff_sims, cutoff_ids = get_top_k_mmr_embeddings(
        query_embedding,
        embeddings,
        similarity_fn=dot,
        similarity_top_k=4,
        mmr_threshold=1,
        similarity_cutoff=0.5,
    )
    assert cutoff_ids == [0, 1]
    assert np.isclose(cutoff_sims[0], 1.0, atol=0.00001)
    assert np.isclose(cutoff_sims[1], 0.9, atol=0.00001)


def test_get_top_k_mmr_embeddings_cutoff_with_embedding_ids() -> None:
    """Custom embedding ids survive cutoff filtering."""
    query_embedding = [1.0, 0.0]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]

    _, ids = get_top_k_mmr_embeddings(
        query_embedding,
        embeddings,
        embedding_ids=["keep", "drop"],
        mmr_threshold=1,
        similarity_cutoff=0.5,
    )
    assert ids == ["keep"]
