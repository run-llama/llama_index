from typing import List

import pytest

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import IndexNode, NodeWithScore, QueryBundle, TextNode


class ZeroScoreRecursiveRetriever(BaseRetriever):
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return [
            NodeWithScore(
                node=IndexNode(
                    index_id="child",
                    text="index",
                    obj=TextNode(text="child"),
                ),
                score=0.0,
            )
        ]

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self._retrieve(query_bundle)


def test_recursive_retrieval_preserves_zero_score() -> None:
    nodes = ZeroScoreRecursiveRetriever().retrieve("query")

    assert nodes[0].score == 0.0


@pytest.mark.asyncio
async def test_async_recursive_retrieval_preserves_zero_score() -> None:
    nodes = await ZeroScoreRecursiveRetriever().aretrieve("query")

    assert nodes[0].score == 0.0
