import asyncio

import pytest

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.retrievers.recursive_retriever import RecursiveRetriever
from llama_index.core.schema import IndexNode, NodeWithScore, QueryBundle, TextNode


class ScoredIndexRetriever(BaseRetriever):
    def __init__(self, score, object_map=None):
        super().__init__(object_map=object_map)
        self.score = score

    def _retrieve(self, query_bundle: QueryBundle):
        return [
            NodeWithScore(
                node=IndexNode(text="parent", index_id="child"), score=self.score
            )
        ]


@pytest.mark.parametrize(
    ("parent_score", "expected_score"),
    [(0.0, 0.0), (0.3, 0.3), (None, 1.0)],
)
def test_recursive_retriever_preserves_zero_score(parent_score, expected_score):
    child = TextNode(text="child")
    retriever = RecursiveRetriever(
        root_id="root",
        retriever_dict={"root": ScoredIndexRetriever(parent_score)},
        node_dict={"child": child},
    )

    result = retriever.retrieve("query")

    assert len(result) == 1
    assert result[0].node.id_ == child.id_
    assert result[0].score == expected_score


@pytest.mark.parametrize("async_retrieve", [False, True])
def test_base_retriever_preserves_zero_score_when_following_index_node(async_retrieve):
    child = TextNode(text="child")
    retriever = ScoredIndexRetriever(0.0, object_map={"child": child})

    result = (
        asyncio.run(retriever.aretrieve("query"))
        if async_retrieve
        else retriever.retrieve("query")
    )

    assert len(result) == 1
    assert result[0].node.id_ == child.id_
    assert result[0].score == 0.0
