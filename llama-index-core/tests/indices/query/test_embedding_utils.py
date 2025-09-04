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
