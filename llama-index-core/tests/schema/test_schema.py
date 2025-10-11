import base64
import httpx
import logging
from io import BytesIO
from pathlib import Path
from unittest import mock

import pytest
from llama_index.core.schema import (
    Document,
    ImageDocument,
    ImageNode,
    MediaResource,
    NodeWithScore,
    ObjectType,
    TextNode,
)


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


@pytest.fixture()
def png_1px_b64() -> bytes:
    return b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


@pytest.fixture()
def png_1px(png_1px_b64) -> bytes:
    return base64.b64decode(png_1px_b64)


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


def test_text_node_with_text_resource():
    tr = MediaResource(text="This is a test")
    text_node = TextNode(text_resource=tr)
    assert text_node.text == "This is a test"

    tr_dict = tr.model_dump()
    text_node = TextNode(text_resource=tr_dict)
    assert text_node.text == "This is a test"


def test_image_node_hash() -> None:
    """Test hash for ImageNode."""
    node = ImageNode(image="base64", image_path="path")
    node2 = ImageNode(image="base64", image_path="path2")
    assert node.hash != node2.hash

    # id's don't count as part of the hash
    node3 = ImageNode(image_url="base64", id_="id")
    node4 = ImageNode(image_url="base64", id_="id2")
    assert node3.hash == node4.hash


def test_image_node_mimetype() -> None:
    node = ImageNode(image_path="path")
    node2 = ImageNode(image_path="path.png")

    assert node.image_mimetype is None
    assert node2.image_mimetype == "image/png"


def test_build_image_node_image_resource() -> None:
    ir = MediaResource(path="my-image.jpg", mimetype=None)
    tr = MediaResource(text="test data")
    node = ImageNode(id_="test_node", image_resource=ir, text_resource=tr)
    assert node.text == "test data"
    assert node.image_mimetype == "image/jpeg"
    assert node.image_path == "my-image.jpg"


def test_build_text_node_text_resource() -> None:
    node = TextNode(id_="test_node", text_resource=MediaResource(text="test data"))
    assert node.text == "test data"


def test_document_init(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        # Legacy init
        doc = Document(doc_id="test")
        assert doc.doc_id == "test"
        assert doc.id_ == "test"
        # Legacy init mixed with new
        doc = Document(id_="test", doc_id="legacy_test")
        assert "'doc_id' is deprecated and 'id_' will be used instead" in caplog.text
        assert doc.id_ == "test"
        caplog.clear()

        # Legacy init
        doc = Document(extra_info={"key": "value"})
        assert doc.metadata == {"key": "value"}
        assert doc.extra_info == {"key": "value"}
        # Legacy init mixed with new
        doc = Document(extra_info={"old_key": "old_value"}, metadata={"key": "value"})
        assert (
            "'extra_info' is deprecated and 'metadata' will be used instead"
            in caplog.text
        )
        assert doc.metadata == {"key": "value"}
        assert doc.extra_info == {"key": "value"}
        caplog.clear()

        # Legacy init
        doc = Document(text="test")
        assert doc.text == "test"
        assert doc.text_resource
        assert doc.text_resource.text == "test"
        # Legacy init mixed with new
        doc = Document(text="legacy_test", text_resource=MediaResource(text="test"))
        assert (
            "'text' is deprecated and 'text_resource' will be used instead"
            in caplog.text
        )
        assert doc.text == "test"
        assert doc.text_resource
        assert doc.text_resource.text == "test"


def test_document_properties():
    doc = Document()
    assert doc.get_type() == ObjectType.DOCUMENT
    doc.doc_id = "test"
    assert doc.id_ == "test"


def test_document_str():
    with mock.patch("llama_index.core.schema.TRUNCATE_LENGTH", 5):
        doc = Document(
            id_="test_id",
            text="Lorem ipsum dolor sit amet, consectetur adipiscing elit",
        )
        assert str(doc) == "Doc ID: test_id\nText: Lo..."


def test_document_legacy_roundtrip():
    origin = Document(id_="test_id", text="this is a test")
    assert origin.model_dump() == {
        "id_": "test_id",
        "embedding": None,
        "metadata": {},
        "excluded_embed_metadata_keys": [],
        "excluded_llm_metadata_keys": [],
        "relationships": {},
        "metadata_template": "{key}: {value}",
        "metadata_separator": "\n",
        "text": "this is a test",
        "text_resource": {
            "embeddings": None,
            "text": "this is a test",
            "mimetype": None,
            "path": None,
            "url": None,
        },
        "image_resource": None,
        "audio_resource": None,
        "video_resource": None,
        "text_template": "{metadata_str}\n\n{content}",
        "class_name": "Document",
    }
    dest = Document(**origin.model_dump())
    assert dest.text == "this is a test"


def test_document_model_dump_exclude():
    doc = Document(id_="test_id", text="this is a test")
    model_dump = doc.model_dump(exclude={"text", "metadata", "relationships"})
    assert "text" not in model_dump
    assert "metadata" not in model_dump
    assert "relationships" not in model_dump
    assert model_dump == {
        "id_": "test_id",
        "embedding": None,
        "excluded_embed_metadata_keys": [],
        "excluded_llm_metadata_keys": [],
        "metadata_template": "{key}: {value}",
        "metadata_separator": "\n",
        "text_resource": {
            "embeddings": None,
            "text": "this is a test",
            "mimetype": None,
            "path": None,
            "url": None,
        },
        "image_resource": None,
        "audio_resource": None,
        "video_resource": None,
        "text_template": "{metadata_str}\n\n{content}",
        "class_name": "Document",
    }


def test_image_document_empty():
    doc = ImageDocument(id_="test")
    assert doc.id_ == "test"
    assert doc.image is None
    assert doc.image_path is None
    assert doc.image_url is None
    assert doc.image_mimetype is None
    assert doc.class_name() == "ImageDocument"


def test_image_document_image():
    doc = ImageDocument(id_="test", image=b"123456")
    assert doc.image == "MTIzNDU2"
    doc.image = "123456789"
    assert doc.image == "MTIzNDU2Nzg5"


def test_image_document_path(tmp_path: Path):
    content = httpx.get(
        "https://astrabert.github.io/hophop-science/images/whale_doing_science.png"
    ).content
    fl_path = tmp_path / "test_image.png"
    fl_path.write_bytes(content)
    doc = ImageDocument(id_="test", image_path=fl_path)
    assert doc.image_path == str(fl_path)
    doc.image_path = str(fl_path.parent)
    assert doc.image_path == str(fl_path.parent)


def test_image_document_url():
    doc = ImageDocument(
        id_="test",
        image_url="https://astrabert.github.io/hophop-science/images/whale_doing_science.png",
    )
    assert (
        doc.image_url
        == "https://astrabert.github.io/hophop-science/images/whale_doing_science.png"
    )
    doc.image_url = "https://foo.org"
    assert doc.image_url == "https://foo.org/"


def test_image_document_mimetype():
    doc = ImageDocument(image=b"123456")
    assert doc.image_mimetype is None
    doc.image_mimetype = "foo"
    assert doc.image_mimetype == "foo"


def test_image_document_embeddings():
    doc = ImageDocument(text="foo")
    assert doc.text_resource is not None
    assert doc.text_embedding is None
    doc.text_embedding = [1.0, 2.0, 3.0]
    assert doc.text_embedding == [1.0, 2.0, 3.0]
    assert doc.text_resource.embeddings == {"dense": [1.0, 2.0, 3.0]}


def test_image_document_path_serialization(tmp_path: Path):
    content = httpx.get(
        "https://astrabert.github.io/hophop-science/images/whale_doing_science.png"
    ).content
    fl_path = tmp_path / "test_image.png"
    fl_path.write_bytes(content)
    doc = ImageDocument(image_path=fl_path)
    assert doc.model_dump()["image_resource"]["path"] == fl_path.__str__()

    new_doc = ImageDocument(**doc.model_dump())
    assert new_doc.image_resource.path == fl_path


def test_image_block_resolve_image(png_1px: bytes, png_1px_b64: bytes):
    doc = ImageDocument()
    assert doc.resolve_image().read() == b""

    doc = ImageDocument(image=png_1px)

    img = doc.resolve_image()
    assert isinstance(img, BytesIO)
    assert img.read() == png_1px

    img = doc.resolve_image(as_base64=True)
    assert isinstance(img, BytesIO)
    assert img.read() == png_1px_b64
