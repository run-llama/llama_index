from tempfile import NamedTemporaryFile
from typing import List

from PIL import Image
import pytest
from llama_index.core.schema import (
    Document,
    ImageDocument,
    ImageNode,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)


@pytest.fixture()
def documents() -> List[Document]:
    """Get text documents."""
    # NOTE: one document for now
    doc_text = (
        "Hello world.\n"
        "This is a test.\n"
        "This is another test.\n"
        "This is a test v2."
    )
    return [Document(text=doc_text)]


@pytest.fixture()
def nodes() -> List[TextNode]:
    """Get text nodes."""
    # NOTE: one document for now
    return [
        TextNode(
            text="Hello world.",
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test doc")
            },
        ),
        TextNode(
            text="This is a test.",
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test doc")
            },
        ),
        TextNode(
            text="This is another test.",
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test doc")
            },
        ),
        TextNode(
            text="This is a test v2.",
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test doc")
            },
        ),
    ]


@pytest.fixture(scope="session")
def image_path_0():
    with NamedTemporaryFile(suffix="image0.png") as f:
        image = Image.new("RGB", (16, 16), color=0)
        image.save(f.name)

        yield f.name


@pytest.fixture(scope="session")
def image_path_1():
    with NamedTemporaryFile(suffix="image1.png") as f:
        image = Image.new("RGB", (16, 16), color=1)
        image.save(f.name)

        yield f.name


@pytest.fixture()
def image_documents(image_path_0, image_path_1) -> List[ImageDocument]:
    """Get image documents."""
    doc_text = (
        "Hello world.\n"
        "This is a test.\n"
        "This is another test.\n"
        "This is a test v2."
    )
    return [
        ImageDocument(text=doc_text[:2], image_path=image_path_0),
        ImageDocument(text=doc_text[2:], image_path=image_path_1),
    ]


@pytest.fixture()
def image_nodes(image_path_0, image_path_1) -> List[ImageNode]:
    """Get image nodes."""
    return [
        ImageNode(
            text="Hello world.",
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test doc 0")
            },
            image=image_path_0,
        ),
        ImageNode(
            text="This is a test.",
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test doc 0")
            },
            image=image_path_0,
        ),
        ImageNode(
            text="This is another test.",
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test doc 1")
            },
            image=image_path_1,
        ),
        ImageNode(
            text="This is a test v2.",
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id="test doc 1")
            },
            image=image_path_1,
        ),
    ]
