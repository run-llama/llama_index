import pytest

from llama_index.core import Document
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    get_child_nodes,
    get_deeper_nodes,
    get_leaf_nodes,
    get_root_nodes,
)
from llama_index.core.schema import NodeRelationship, TextNode

ROOT_NODES_LEN = 1
CHILDREN_NODES_LEN = 3
GRAND_CHILDREN_NODES_LEN = 7


@pytest.fixture(scope="module")
def nodes() -> list:
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[512, 128, 64],
        chunk_overlap=10,
    )
    return node_parser.get_nodes_from_documents([Document.example()])


def test_get_root_nodes(nodes: list) -> None:
    root_nodes = get_root_nodes(nodes)
    assert len(root_nodes) == ROOT_NODES_LEN


def test_get_root_nodes_empty(nodes: list) -> None:
    root_nodes = get_root_nodes(get_leaf_nodes(nodes))
    assert root_nodes == []


def test_get_leaf_nodes(nodes: list) -> None:
    leaf_nodes = get_leaf_nodes(nodes)
    assert len(leaf_nodes) == GRAND_CHILDREN_NODES_LEN


def test_get_child_nodes(nodes: list) -> None:
    child_nodes = get_child_nodes(get_root_nodes(nodes), all_nodes=nodes)
    assert len(child_nodes) == CHILDREN_NODES_LEN


def test_get_deeper_nodes(nodes: list) -> None:
    deep_nodes = get_deeper_nodes(nodes, depth=0)
    assert deep_nodes == get_root_nodes(nodes)

    deep_nodes = get_deeper_nodes(nodes, depth=1)
    assert deep_nodes == get_child_nodes(get_root_nodes(nodes), nodes)

    deep_nodes = get_deeper_nodes(nodes, depth=2)
    assert deep_nodes == get_leaf_nodes(nodes)

    deep_nodes = get_deeper_nodes(nodes, depth=2)
    assert deep_nodes == get_child_nodes(
        get_child_nodes(get_root_nodes(nodes), nodes), nodes
    )


def test_get_deeper_nodes_with_no_root_nodes(nodes: list) -> None:
    with pytest.raises(ValueError, match="There is no*"):
        get_deeper_nodes(get_leaf_nodes(nodes))


def test_get_deeper_nodes_with_negative_depth(nodes: list) -> None:
    with pytest.raises(ValueError, match="Depth cannot be*"):
        get_deeper_nodes(nodes, -1)


def test_get_child_nodes_preserves_order_and_deduplicates_shared_children() -> None:
    parent_a = TextNode(id_="p_a", text="parent a")
    parent_b = TextNode(id_="p_b", text="parent b")
    child_1 = TextNode(id_="c_1", text="child 1")
    child_2 = TextNode(id_="c_2", text="child 2")
    child_3 = TextNode(id_="c_3", text="child 3")
    unrelated = TextNode(id_="u_1", text="unrelated")

    parent_a.relationships[NodeRelationship.CHILD] = [
        child_2.as_related_node_info(),
        child_1.as_related_node_info(),
    ]
    parent_b.relationships[NodeRelationship.CHILD] = [
        child_2.as_related_node_info(),
        child_3.as_related_node_info(),
    ]

    all_nodes = [parent_a, parent_b, child_1, unrelated, child_2, child_3]
    result = get_child_nodes([parent_a, parent_b], all_nodes=all_nodes)
    assert [n.node_id for n in result] == ["c_1", "c_2", "c_3"]
