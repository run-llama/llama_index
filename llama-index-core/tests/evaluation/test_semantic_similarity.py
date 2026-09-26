import math
from typing import Any, Dict, List

import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.evaluation import SemanticSimilarityEvaluator


class FixedEmbedding(BaseEmbedding):
    """
    Embedding model that returns a pre-set vector for each known text.

    Lets a test pin the exact cosine similarity between two texts without
    depending on a real embedding model.
    """

    text_to_vector: Dict[str, List[float]]

    def __init__(self, text_to_vector: Dict[str, List[float]], **kwargs: Any) -> None:
        super().__init__(text_to_vector=text_to_vector, **kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "FixedEmbedding"

    def _get_vector(self, text: str) -> List[float]:
        return self.text_to_vector[text]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_vector(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_vector(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_vector(text)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_vector(text)


def _vector_at_cosine(target_cosine: float) -> List[float]:
    """A 2D unit vector whose cosine similarity to [1, 0] is target_cosine."""
    return [target_cosine, math.sqrt(1 - target_cosine**2)]


@pytest.mark.asyncio
async def test_reversed_meaning_can_pass_at_default_threshold() -> None:
    """
    Cosine similarity does not track correctness: a wording-preserving
    reversal of the reference's meaning can still score above the default
    0.8 threshold. See https://github.com/run-llama/llama_index/issues/22729.
    """
    reference = (
        "Withhold the study drug from any participant who reports chest tightness."
    )
    response = (
        "Administer the study drug to any participant who reports chest tightness."
    )

    embed_model = FixedEmbedding(
        text_to_vector={
            reference: [1.0, 0.0],
            response: _vector_at_cosine(0.96),
        }
    )
    evaluator = SemanticSimilarityEvaluator(embed_model=embed_model)

    result = await evaluator.aevaluate(response=response, reference=reference)

    assert result.score == pytest.approx(0.96)
    assert result.passing is True


@pytest.mark.asyncio
async def test_faithful_low_overlap_restatement_can_fail_at_default_threshold() -> None:
    """
    The mirror-image blind spot: a correct restatement that shares little
    wording with the reference can score below the default 0.8 threshold.
    See https://github.com/run-llama/llama_index/issues/22729.
    """
    reference = "Retry the request at most three times."
    response = "Give the call up to three attempts, then stop."

    embed_model = FixedEmbedding(
        text_to_vector={
            reference: [1.0, 0.0],
            response: _vector_at_cosine(0.76),
        }
    )
    evaluator = SemanticSimilarityEvaluator(embed_model=embed_model)

    result = await evaluator.aevaluate(response=response, reference=reference)

    assert result.score == pytest.approx(0.76)
    assert result.passing is False
