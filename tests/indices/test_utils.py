from gpt_index.indices.utils import get_text_from_nodes
from gpt_index.indices.data_structs import Node
from typing import List


def test_get_text_from_nodes() -> str:
    """Get text from nodes. Used by tree-structured indices."""

    node1 = Node(
        "This is a test. Hello my name is John.", 0, set()
    )
    node2 = Node(
        "This is another test. Hello world!", 1, set()
    )
    node_list = [node1, node2]
    response = get_text_from_nodes(node_list)
    assert response == (
        "This is a test. Hello my name is John.\n"
        "This is another test. Hello world!"
    )
