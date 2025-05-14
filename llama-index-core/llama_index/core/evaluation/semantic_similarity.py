from typing import Any, Callable, Optional, Sequence

from llama_index.core.base.embeddings.base import (
    BaseEmbedding,
    SimilarityMode,
    similarity,
)
from llama_index.core.evaluation.base import BaseEvaluator, EvaluationResult
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.settings import Settings


class SemanticSimilarityEvaluator(BaseEvaluator):
    """
    Embedding similarity evaluator.

    Evaluate the quality of a question answering system by
    comparing the similarity between embeddings of the generated answer
    and the reference answer.

    Inspired by this paper:
    - Semantic Answer Similarity for Evaluating Question Answering Models
        https://arxiv.org/pdf/2108.06130.pdf

    Args:
        similarity_threshold (float): Embedding similarity threshold for "passing".
            Defaults to 0.8.

    """

    def __init__(
        self,
        embed_model: Optional[BaseEmbedding] = None,
        similarity_fn: Optional[Callable[..., float]] = None,
        similarity_mode: Optional[SimilarityMode] = None,
        similarity_threshold: float = 0.8,
    ) -> None:
        self._embed_model = embed_model or Settings.embed_model

        if similarity_fn is None:
            similarity_mode = similarity_mode or SimilarityMode.DEFAULT
            self._similarity_fn = lambda x, y: similarity(x, y, mode=similarity_mode)
        else:
            if similarity_mode is not None:
                raise ValueError(
                    "Cannot specify both similarity_fn and similarity_mode"
                )
            self._similarity_fn = similarity_fn

        self._similarity_threshold = similarity_threshold

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""

    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        del query, contexts, kwargs  # Unused

        if response is None or reference is None:
            raise ValueError("Must specify both response and reference")

        response_embedding = await self._embed_model.aget_text_embedding(response)
        reference_embedding = await self._embed_model.aget_text_embedding(reference)

        similarity_score = self._similarity_fn(response_embedding, reference_embedding)
        passing = similarity_score >= self._similarity_threshold
        return EvaluationResult(
            score=similarity_score,
            passing=passing,
            feedback=f"Similarity score: {similarity_score}",
        )
