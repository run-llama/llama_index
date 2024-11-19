from typing import Any

import pytest
from llama_index.core.schema import (
    BaseNode,
    MetadataMode,
    NodeRelationship,
    ObjectType,
    RelatedNodeInfo,
)


@pytest.fixture()
def MyNode():
    class MyNode(BaseNode):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

        @classmethod
        def get_type(cls):
            return ObjectType.TEXT

        def get_content(self, metadata_mode: MetadataMode = MetadataMode.ALL) -> str:
            return "Test content"

        def set_content(self, value: Any) -> None:
            return super().set_content(value)

        @property
        def hash(self) -> str:
            return super().hash

    return MyNode


def test_get_metadata_str(MyNode):
    metadata = {
        "key": "value",
        "forbidden": "true",
    }
    excluded = ["forbidden"]
    node = MyNode(
        metadata=metadata,
        excluded_llm_metadata_keys=excluded,
        excluded_embed_metadata_keys=excluded,
    )
    assert node.get_metadata_str(MetadataMode.NONE) == ""
    assert node.get_metadata_str(MetadataMode.LLM) == "key: value"
    assert node.get_metadata_str(MetadataMode.EMBED) == "key: value"


def test_node_id(MyNode):
    n = MyNode()
    n.node_id = "this"
    assert n.node_id == "this"


def test_source_node(MyNode):
    n1 = MyNode()
    n2 = MyNode(
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=n1.node_id)}
    )
    assert n2.source_node.hash == n1.hash
    assert n1.source_node is None

    with pytest.raises(
        ValueError, match="Source object must be a single RelatedNodeInfo object"
    ):
        n3 = MyNode(
            relationships={
                NodeRelationship.SOURCE: [RelatedNodeInfo(node_id=n1.node_id)]
            }
        )
        n3.source_node


def test_prev_node(MyNode):
    n1 = MyNode()
    n2 = MyNode(
        relationships={NodeRelationship.PREVIOUS: RelatedNodeInfo(node_id=n1.node_id)}
    )
    assert n2.prev_node.hash == n1.hash
    assert n1.prev_node is None

    with pytest.raises(
        ValueError, match="Previous object must be a single RelatedNodeInfo object"
    ):
        n3 = MyNode(
            relationships={
                NodeRelationship.PREVIOUS: [RelatedNodeInfo(node_id=n1.node_id)]
            }
        )
        n3.prev_node


def test_next_node(MyNode):
    n1 = MyNode()
    n2 = MyNode(
        relationships={NodeRelationship.NEXT: RelatedNodeInfo(node_id=n1.node_id)}
    )
    assert n2.next_node.hash == n1.hash
    assert n1.next_node is None

    with pytest.raises(
        ValueError, match="Next object must be a single RelatedNodeInfo object"
    ):
        n3 = MyNode(
            relationships={NodeRelationship.NEXT: [RelatedNodeInfo(node_id=n1.node_id)]}
        )
        n3.next_node


def test_parent_node(MyNode):
    n1 = MyNode()
    n2 = MyNode(
        relationships={NodeRelationship.PARENT: RelatedNodeInfo(node_id=n1.node_id)}
    )
    assert n2.parent_node.hash == n1.hash
    assert n1.parent_node is None

    with pytest.raises(
        ValueError, match="Parent object must be a single RelatedNodeInfo object"
    ):
        n3 = MyNode(
            relationships={
                NodeRelationship.PARENT: [RelatedNodeInfo(node_id=n1.node_id)]
            }
        )
        n3.parent_node


def test_child_node(MyNode):
    n1 = MyNode()
    n2 = MyNode(
        relationships={NodeRelationship.CHILD: [RelatedNodeInfo(node_id=n1.node_id)]}
    )
    assert n2.child_nodes[0].hash == n1.hash
    assert n1.child_nodes is None

    with pytest.raises(
        ValueError, match="Child objects must be a list of RelatedNodeInfo objects"
    ):
        n3 = MyNode(
            relationships={NodeRelationship.CHILD: RelatedNodeInfo(node_id=n1.node_id)}
        )
        n3.child_nodes


def test___str__(MyNode):
    n = MyNode()
    n.node_id = "test_node"
    assert str(n) == "Node ID: test_node\nText: Test content"


def test_get_embedding(MyNode):
    n = MyNode()
    with pytest.raises(ValueError, match="embedding not set."):
        n.get_embedding()
    n.embedding = [0.0, 0.0]
    assert n.get_embedding() == [0.0, 0.0]


def test_as_related_node_info(MyNode):
    n = MyNode(id_="test_node")
    assert n.as_related_node_info().node_id == "test_node"
