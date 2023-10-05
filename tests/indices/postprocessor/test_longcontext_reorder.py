from typing import List

from llama_index.indices.postprocessor.node import LongContextReorder
from llama_index.schema import Node, NodeWithScore


def test_long_context_reorder() -> None:
    nodes = [
        NodeWithScore(node=Node(text="text"), score=0.7),
        NodeWithScore(node=Node(text="text"), score=0.8),
        NodeWithScore(node=Node(text="text"), score=1.0),
        NodeWithScore(node=Node(text="text"), score=0.2),
        NodeWithScore(node=Node(text="text"), score=0.9),
        NodeWithScore(node=Node(text="text"), score=1.5),
        NodeWithScore(node=Node(text="text"), score=0.1),
        NodeWithScore(node=Node(text="text"), score=1.6),
        NodeWithScore(node=Node(text="text"), score=3.0),
        NodeWithScore(node=Node(text="text"), score=0.4),
    ]
    ordered_nodes: List[NodeWithScore] = sorted(
        nodes, key=lambda x: x.score if x.score is not None else 0, reverse=True
    )
    expected_scores_at_tails = [n.score for n in ordered_nodes[:4]]
    lcr = LongContextReorder()
    filtered_nodes = lcr.postprocess_nodes(nodes)
    nodes_lost_in_the_middle = [n.score for n in filtered_nodes[3:-2]]
    assert set(expected_scores_at_tails).intersection(nodes_lost_in_the_middle) == set()
