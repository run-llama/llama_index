import pytest

from llama_index.core.postprocessor import PrevNextNodePostprocessor
from llama_index.core.schema import NodeRelationship, NodeWithScore, TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore


def _make_store():
    nodes = [TextNode(text=f"Node {i}", id_=str(i)) for i in range(3)]
    for current, nxt in zip(nodes, nodes[1:]):
        current.relationships[NodeRelationship.NEXT] = nxt.as_related_node_info()
        nxt.relationships[NodeRelationship.PREVIOUS] = current.as_related_node_info()

    store = SimpleDocumentStore()
    store.add_documents(nodes)
    return store, nodes


@pytest.mark.parametrize(
    ("mode", "retrieved_ids"),
    [
        ("next", ["1", "0"]),
        ("previous", ["0", "1"]),
        ("both", ["1", "0"]),
    ],
)
def test_prev_next_preserves_retrieved_scores(mode, retrieved_ids):
    store, nodes = _make_store()
    scores = {"0": 0.8, "1": 0.9}
    retrieved = [
        NodeWithScore(node=nodes[int(node_id)], score=scores[node_id])
        for node_id in retrieved_ids
    ]

    result = PrevNextNodePostprocessor(
        docstore=store, mode=mode, num_nodes=1
    ).postprocess_nodes(retrieved)

    result_scores = {node.node.node_id: node.score for node in result}
    assert result_scores["0"] == 0.8
    assert result_scores["1"] == 0.9


def test_prev_next_preserves_zero_score():
    store, nodes = _make_store()
    retrieved = [
        NodeWithScore(node=nodes[1], score=0.0),
        NodeWithScore(node=nodes[0], score=0.8),
    ]

    result = PrevNextNodePostprocessor(
        docstore=store, mode="next", num_nodes=1
    ).postprocess_nodes(retrieved)

    result_scores = {node.node.node_id: node.score for node in result}
    assert result_scores["1"] == 0.0
    assert result_scores["0"] == 0.8


def test_prev_next_added_neighbors_remain_unscored():
    store, nodes = _make_store()
    retrieved = [NodeWithScore(node=nodes[0], score=0.8)]

    result = PrevNextNodePostprocessor(
        docstore=store, mode="next", num_nodes=2
    ).postprocess_nodes(retrieved)

    result_scores = {node.node.node_id: node.score for node in result}
    assert result_scores["0"] == 0.8
    assert result_scores["1"] is None
    assert result_scores["2"] is None
