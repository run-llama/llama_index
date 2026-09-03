from typing import List

import pytest
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.evaluation.retrieval.evaluator import (
    MultiModalRetrieverEvaluator,
)
from llama_index.core.schema import ImageNode, NodeWithScore, QueryBundle, TextNode


class MockMultiModalRetriever(BaseRetriever):
    """Retriever returning one text node and one image node."""

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return [
            NodeWithScore(node=TextNode(id_="text-1", text="a text chunk")),
            NodeWithScore(
                node=ImageNode(id_="image-1", text="a caption", image="fake-b64")
            ),
        ]


@pytest.mark.asyncio
async def test_multi_modal_evaluator_keeps_the_two_modes_separate() -> None:
    """
    `ImageNode` subclasses `TextNode`, so an image node satisfies both
    `isinstance` checks. Evaluating in text mode used to score the retriever on
    image nodes as well, which is the opposite of what this evaluator is for.
    """
    evaluator = MultiModalRetrieverEvaluator.from_metric_names(
        ["mrr", "hit_rate"], retriever=MockMultiModalRetriever()
    )

    text_ids, _ = await evaluator._aget_retrieved_ids_and_texts("query", "text")
    image_ids, _ = await evaluator._aget_retrieved_ids_and_texts("query", "image")

    assert text_ids == ["text-1"]
    assert image_ids == ["image-1"]
