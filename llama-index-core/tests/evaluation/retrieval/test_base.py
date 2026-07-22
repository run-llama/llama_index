"""Tests for the retrieval-evaluation base contract."""

from typing import List, Tuple

import pytest

from llama_index.core.evaluation.retrieval.base import (
    BaseRetrievalEvaluator,
    RetrievalEvalMode,
)


class _MinimalRetrievalEvaluator(BaseRetrievalEvaluator):
    """
    Retrieval evaluator that delegates the async path to the base.

    ``_aget_retrieved_ids_and_texts`` is ``@abstractmethod`` with a
    ``raise NotImplementedError`` body. Here we override it to call
    ``super()._aget_retrieved_ids_and_texts(...)`` so the base body is
    exercised and verified to raise descriptively.
    """

    def _get_retrieved_ids_and_texts(
        self, query: str, mode: RetrievalEvalMode = RetrievalEvalMode.TEXT
    ) -> Tuple[List[str], List[str]]:
        return [], []

    async def _aget_retrieved_ids_and_texts(
        self, query: str, mode: RetrievalEvalMode = RetrievalEvalMode.TEXT
    ) -> Tuple[List[str], List[str]]:
        return await super()._aget_retrieved_ids_and_texts(query, mode)


def test_from_str_unknown_label_raises_descriptive() -> None:
    """RetrievalEvalMode.from_str must raise with a message listing valid labels."""
    with pytest.raises(NotImplementedError, match="supported labels"):
        RetrievalEvalMode.from_str("audio")


@pytest.mark.asyncio
async def test_aget_retrieved_ids_and_texts_raises_descriptive() -> None:
    """BaseRetrievalEvaluator._aget_retrieved_ids_and_texts must raise descriptively."""
    evaluator = _MinimalRetrievalEvaluator(metrics=[])
    with pytest.raises(NotImplementedError, match="_aget_retrieved_ids_and_texts"):
        await evaluator._aget_retrieved_ids_and_texts("q")
