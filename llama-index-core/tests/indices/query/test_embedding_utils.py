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


def test_get_top_k_embeddings_top_k_zero() -> None:
    """
    An explicit similarity_top_k of 0 must return no results.

    `similarity_top_k` is optional and defaults to None, which means "no
    limit". Because 0 is falsy, `if similarity_top_k and ...` conflated the two
    and returned every embedding when a caller explicitly asked for none.
    """
    query_embedding = [1.0, 0.0]
    embeddings = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]

    _, result_ids = get_top_k_embeddings(
        query_embedding, embeddings, similarity_top_k=0
    )
    assert result_ids == []

    # None still means "no limit".
    _, unset_ids = get_top_k_embeddings(query_embedding, embeddings)
    assert len(unset_ids) == len(embeddings)

    # An ordinary limit is unaffected.
    _, top_2_ids = get_top_k_embeddings(query_embedding, embeddings, similarity_top_k=2)
    assert len(top_2_ids) == 2


def test_get_top_k_mmr_embeddings_top_k_zero() -> None:
    """An explicit similarity_top_k of 0 must return no results under MMR too."""
    query_embedding = [1.0, 0.0]
    embeddings = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]

    _, result_ids = get_top_k_mmr_embeddings(
        query_embedding, embeddings, similarity_top_k=0
    )
    assert result_ids == []

    # None still means "no limit".
    _, unset_ids = get_top_k_mmr_embeddings(query_embedding, embeddings)
    assert len(unset_ids) == len(embeddings)

    # An ordinary limit is unaffected.
    _, top_2_ids = get_top_k_mmr_embeddings(
        query_embedding, embeddings, similarity_top_k=2
    )
    assert len(top_2_ids) == 2
