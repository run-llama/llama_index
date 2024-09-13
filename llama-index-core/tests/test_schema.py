import pytest
from llama_index.core.schema import NodeWithScore, TextNode, ImageNode


@pytest.fixture()
def text_node() -> TextNode:
    return TextNode(
        text="hello world",
        metadata={"foo": "bar"},
        embedding=[0.1, 0.2, 0.3],
    )


@pytest.fixture()
def node_with_score(text_node: TextNode) -> NodeWithScore:
    return NodeWithScore(
        node=text_node,
        score=0.5,
    )


def test_node_with_score_passthrough(node_with_score: NodeWithScore) -> None:
    _ = node_with_score.id_
    _ = node_with_score.node_id
    _ = node_with_score.text
    _ = node_with_score.metadata
    _ = node_with_score.embedding
    _ = node_with_score.get_text()
    _ = node_with_score.get_content()
    _ = node_with_score.get_embedding()


def test_text_node_hash() -> None:
    node = TextNode(text="hello", metadata={"foo": "bar"})
    assert (
        node.hash == "aa158bf3388f103cef4bd85b2ca93f343ad8f5e50f58ae4141a35d75a2f21fb0"
    )
    node.set_content("world")
    assert (
        node.hash == "ce6a3cefc3451ecb1ff41ec41a7d7e24354983520d8b2d6f5447be0b6b9b6b99"
    )

    node.text = "new"
    assert (
        node.hash == "bef8ff82498c9aa7d9f9751f441da9a1a1c4e9941bd03c57caa4a602cd5cadd0"
    )
    node2 = TextNode(text="new", metadata={"foo": "bar"})
    assert node2.hash == node.hash
    node3 = TextNode(text="new", metadata={"foo": "baz"})
    assert node3.hash != node.hash


def test_image_node_hash() -> None:
    """Test hash for ImageNode."""
    node = ImageNode(image="base64", image_path="path")
    node2 = ImageNode(image="base64", image_path="path2")
    assert node.hash != node2.hash

    # id's don't count as part of the hash
    node3 = ImageNode(image_url="base64", id_="id")
    node4 = ImageNode(image_url="base64", id_="id2")
    assert node3.hash == node4.hash
