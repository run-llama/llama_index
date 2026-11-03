from typing import List, Optional
from unittest.mock import Mock

import pytest

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import Response
from llama_index.core.retrievers.recursive_retriever import RecursiveRetriever
from llama_index.core.schema import IndexNode, NodeWithScore, QueryBundle, TextNode


class StaticRetriever(BaseRetriever):
    def __init__(self, nodes: List[NodeWithScore]) -> None:
        super().__init__()
        self._nodes = nodes

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self._nodes


@pytest.mark.parametrize(
    ("first_score", "second_score", "expected_text", "expected_score"),
    [
        (0.2, 0.9, "second", 0.9),
        (0.9, 0.2, "first", 0.9),
        (0.5, 0.5, "first", 0.5),
        (None, 0.8, "second", 0.8),
        (0.8, None, "first", 0.8),
        (None, None, "first", None),
        (0.0, None, "first", 0.0),
        (-0.5, None, "second", None),
    ],
)
def test_duplicate_nodes_keep_highest_score_and_first_position(
    first_score: Optional[float],
    second_score: Optional[float],
    expected_text: str,
    expected_score: Optional[float],
) -> None:
    first = NodeWithScore(node=TextNode(id_="shared", text="first"), score=first_score)
    unique = NodeWithScore(node=TextNode(id_="unique", text="unique"), score=0.3)
    second = NodeWithScore(
        node=TextNode(id_="shared", text="second"), score=second_score
    )
    root = StaticRetriever(
        [
            NodeWithScore(node=IndexNode(index_id="first", text="first")),
            NodeWithScore(node=IndexNode(index_id="second", text="second")),
        ]
    )
    retriever = RecursiveRetriever(
        root_id="root",
        retriever_dict={
            "root": root,
            "first": StaticRetriever([first, unique]),
            "second": StaticRetriever([second]),
        },
    )

    result = retriever.retrieve("query")

    assert [node.node.id_ for node in result] == ["shared", "unique"]
    assert result[0] is (first if expected_text == "first" else second)
    assert result[0].node.text == expected_text
    assert result[0].score == expected_score
    assert first.score == first_score
    assert second.score == second_score
    assert result[1] is unique
    assert result[1].node.text == "unique"
    assert result[1].score == 0.3


def test_additional_nodes_keep_highest_score_and_first_position() -> None:
    first = NodeWithScore(node=TextNode(id_="shared", text="first"), score=0.2)
    unique = NodeWithScore(node=TextNode(id_="unique", text="unique"), score=0.3)
    second = NodeWithScore(node=TextNode(id_="shared", text="second"), score=0.9)
    first_engine = Mock(spec=BaseQueryEngine)
    first_engine.query.return_value = Response("first response", [first, unique])
    second_engine = Mock(spec=BaseQueryEngine)
    second_engine.query.return_value = Response("second response", [second])
    root = StaticRetriever(
        [
            NodeWithScore(node=IndexNode(index_id="first", text="first")),
            NodeWithScore(node=IndexNode(index_id="second", text="second")),
        ]
    )
    retriever = RecursiveRetriever(
        root_id="root",
        retriever_dict={"root": root},
        query_engine_dict={"first": first_engine, "second": second_engine},
    )

    retrieved, additional = retriever.retrieve_all(QueryBundle("query"))

    assert len(retrieved) == 2
    assert [node.node.id_ for node in additional] == ["shared", "unique"]
    assert additional[0] is second
    assert additional[1] is unique
    assert first.score == 0.2
    assert second.score == 0.9
