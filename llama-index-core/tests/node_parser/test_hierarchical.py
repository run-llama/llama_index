import pytest

from llama_index.core import Document
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    get_child_nodes,
    get_deeper_nodes,
    get_leaf_nodes,
    get_root_nodes,
)

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
