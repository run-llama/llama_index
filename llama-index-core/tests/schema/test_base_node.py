from typing import Any

import pytest

from llama_index.core.base.llms.types import BaseContentBlock, TextBlock
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

        def get_content_blocks(
            self, metadata_mode: MetadataMode = MetadataMode.ALL
        ) -> list[BaseContentBlock]:
            return [TextBlock(text="Test content")]

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


def test_get_embedding_with_key(MyNode):
    n = MyNode()
    n.set_embedding("default", [1.0, 2.0, 3.0])
    assert n.get_embedding() == [1.0, 2.0, 3.0]
    assert n.get_embedding("default") == [1.0, 2.0, 3.0]
    assert n.embedding == [1.0, 2.0, 3.0]
    n.set_embedding("sparse", {"idx_1": 0.5, "idx_2": 0.3})
    assert n.get_embedding("sparse") == {"idx_1": 0.5, "idx_2": 0.3}
    with pytest.raises(ValueError, match="embedding 'missing' not set."):
        n.get_embedding("missing")


def test_set_embedding_syncs_default_to_embedding_field(MyNode):
    n = MyNode()
    n.set_embedding("default", [1.0, 2.0])
    assert n.embedding == [1.0, 2.0]
    assert n.get_embedding() == [1.0, 2.0]
    n.set_embedding("default", [3.0, 4.0])
    assert n.embedding == [3.0, 4.0]


def test_set_embeddings(MyNode):
    n = MyNode()
    n.set_embeddings({"dense": [1.0, 2.0], "sparse": [0.1, 0.2]})
    assert n.get_embedding("dense") == [1.0, 2.0]
    assert n.get_embedding("sparse") == [0.1, 0.2]
    n.set_embeddings({"extra": [9.0]})
    assert n.get_embedding("extra") == [9.0]


def test_get_embeddings(MyNode):
    n = MyNode()
    n.embedding = [0.0, 1.0]
    all_emb = n.get_embeddings()
    assert "default" in all_emb
    assert all_emb["default"] == [0.0, 1.0]
    n.set_embedding("sparse", [0.5])
    all_emb = n.get_embeddings()
    assert all_emb.get("sparse") == [0.5]
    subset = n.get_embeddings(keys=["default", "sparse"])
    assert subset["default"] == [0.0, 1.0]
    assert subset["sparse"] == [0.5]
    empty_subset = n.get_embeddings(keys=["nonexistent"])
    assert empty_subset == {}


def test_as_related_node_info(MyNode):
    n = MyNode(id_="test_node")
    assert n.as_related_node_info().node_id == "test_node"


def test_construction_with_embedding_syncs_embeddings_dict(MyNode):
    n = MyNode(embedding=[1.0, 2.0])
    assert n.embeddings.get("default") == [1.0, 2.0]
    assert n.get_embedding() == [1.0, 2.0]


def test_direct_embedding_assignment_syncs_embeddings_dict(MyNode):
    n = MyNode()
    n.embedding = [1.0, 2.0, 3.0]
    assert n.embeddings.get("default") == [1.0, 2.0, 3.0]
    assert n.get_embedding() == [1.0, 2.0, 3.0]
    assert n.get_embedding("default") == [1.0, 2.0, 3.0]


def test_direct_embedding_assignment_overwrites_embeddings_dict(MyNode):
    n = MyNode()
    n.set_embedding("default", [1.0, 2.0])
    n.embedding = [3.0, 4.0]
    assert n.embeddings.get("default") == [3.0, 4.0]
    assert n.get_embedding() == [3.0, 4.0]


def test_set_embeddings_with_kwargs(MyNode):
    n = MyNode()
    n.set_embeddings(dense=[1.0, 2.0], sparse=[0.1, 0.2])
    assert n.get_embedding("dense") == [1.0, 2.0]
    assert n.get_embedding("sparse") == [0.1, 0.2]


def test_set_embedding_sparse_as_default_key(MyNode):
    n = MyNode()
    n.set_embedding("default", {"idx_0": 0.9, "idx_5": 0.3})
    assert n.embeddings["default"] == {"idx_0": 0.9, "idx_5": 0.3}
    assert n.embedding is None
    result = n.get_embedding()
    assert result == {"idx_0": 0.9, "idx_5": 0.3}


def test_clearing_embedding_behavior(MyNode):
    # Case 1: Dense embedding (Legacy behavior)
    n = MyNode(embedding=[1.0, 2.0])
    assert n.embeddings["default"] == [1.0, 2.0]

    # Set to None - should clear it
    n.embedding = None

    assert "default" not in n.embeddings
    with pytest.raises(ValueError, match="embedding not set."):
        n.get_embedding()

    # Case 2: Sparse embedding (New behavior)
    n2 = MyNode()
    n2.set_embedding("default", {"idx": 1})
    # n2.embedding is None because it's a dict
    assert n2.embedding is None

    # Set to None - should NOT clear sparse embedding
    n2.embedding = None

    assert n2.embeddings.get("default") == {"idx": 1}


def test_overwrite_dense_with_sparse_default(MyNode):
    # Initial state: dense embedding
    n = MyNode(embedding=[1.0, 2.0])
    assert n.get_embedding() == [1.0, 2.0]

    # Overwrite default with sparse
    n.set_embedding("default", {"idx": 1})

    # Expectation: get_embedding() returns the new sparse embedding
    # and n.embedding is cleared (because it can't hold a dict)
    assert n.embeddings["default"] == {"idx": 1}
    assert n.embedding is None
    assert n.get_embedding() == {"idx": 1}


def test_serialization_round_trip(MyNode):
    from llama_index.core.schema import TextNode

    n = TextNode(text="hello", id_="abc")
    n.set_embedding("default", [1.0, 2.0])
    n.set_embedding("sparse", {"a": 0.5})
    d = n.to_dict()
    assert d["embeddings"]["default"] == [1.0, 2.0]
    assert d["embeddings"]["sparse"] == {"a": 0.5}
    n2 = TextNode.from_dict(d)
    assert n2.get_embedding("default") == [1.0, 2.0]
    assert n2.get_embedding("sparse") == {"a": 0.5}
    assert n2.embedding == [1.0, 2.0]
