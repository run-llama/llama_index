from typing import Any

import pytest
from llama_index.core.schema import (
    BaseNode,
    Document,
    MediaResource,
    Node,
    NodeRelationship,
    TextNode,
    ImageNode,
    IndexNode,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)


@pytest.fixture
def source_node():
    return Document(id_="source_node", text="Hello, world!")


@pytest.fixture
def text_node(source_node: Document):
    return TextNode(
        id_="text_node", text="Hello, world!", ref_doc_id=source_node.ref_doc_id
    )


@pytest.fixture
def image_node():
    return ImageNode(id_="image_node", image="tests/data/images/dog.jpg")


@pytest.fixture
def index_node():
    return IndexNode(id_="index_node", text="Hello, world!", index_id="123")


@pytest.fixture
def multimedia_node():
    return Node(
        id_="multimedia_node", text_resource=MediaResource(text="Hello, world!")
    )


def test_text_node_serdes(text_node: TextNode, source_node: Document):
    text_node.relationships[NodeRelationship.SOURCE] = (
        source_node.as_related_node_info()
    )
    serialized_node = node_to_metadata_dict(text_node)
    assert serialized_node["document_id"] == source_node.node_id
    assert serialized_node["ref_doc_id"] == source_node.node_id
    assert serialized_node["doc_id"] == source_node.node_id
    assert "text_node" in serialized_node["_node_content"]
    assert serialized_node["_node_type"] == text_node.class_name()
    deserialized_node = metadata_dict_to_node(serialized_node)
    assert isinstance(deserialized_node, TextNode)
    assert deserialized_node.text == text_node.text


def test_image_node_serdes(image_node: ImageNode):
    serialized_node = node_to_metadata_dict(image_node)
    assert "image_node" in serialized_node["_node_content"]
    assert serialized_node["_node_type"] == image_node.class_name()
    deserialized_node = metadata_dict_to_node(serialized_node)
    assert isinstance(deserialized_node, ImageNode)
    assert deserialized_node.image == image_node.image


def test_index_node_serdes(index_node: IndexNode):
    serialized_node = node_to_metadata_dict(index_node)
    assert "index_node" in serialized_node["_node_content"]
    assert serialized_node["_node_type"] == index_node.class_name()
    deserialized_node = metadata_dict_to_node(serialized_node)
    assert isinstance(deserialized_node, IndexNode)
    assert deserialized_node.text == index_node.text
    assert deserialized_node.index_id == index_node.index_id


def test_multimedia_node_serdes(multimedia_node: Node):
    serialized_node: dict[str, Any] = node_to_metadata_dict(multimedia_node)
    assert "multimedia_node" in serialized_node["_node_content"]
    assert serialized_node["_node_type"] == multimedia_node.class_name()
    deserialized_node: BaseNode = metadata_dict_to_node(serialized_node)

    assert isinstance(deserialized_node, Node)
    assert deserialized_node.text_resource is not None
    assert isinstance(deserialized_node.text_resource, MediaResource)
    assert deserialized_node.text_resource.text is not None
    assert deserialized_node.text_resource.text == multimedia_node.text_resource.text


def test_flat_metadata_serdes(text_node: TextNode):
    text_node.metadata = {"key": {"subkey": "value"}}
    with pytest.raises(ValueError):
        node_to_metadata_dict(text_node, flat_metadata=True)
