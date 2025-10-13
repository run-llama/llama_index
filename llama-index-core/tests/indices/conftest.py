from typing import List

import pytest
from llama_index.core.schema import (
    Document,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)


@pytest.fixture()
def documents() -> List[Document]:
    """Get documents."""
    # NOTE: one document for now
    doc_text = (
        "Hello world.\nThis is a test.\nThis is another test.\nThis is a test v2."
    )
    return [Document(text=doc_text)]


@pytest.fixture()
def nodes() -> List[TextNode]:
    """Get documents."""
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
