from typing import List
import pytest

from llama_index.schema import Document
from llama_index.schema import NodeRelationship, RelatedNodeInfo, TextNode


@pytest.fixture
def documents() -> List[Document]:
    """Get documents."""
    # NOTE: one document for now
    doc_text = (
        "Hello world.\n"
        "This is a test.\n"
        "This is another test.\n"
        "This is a test v2."
    )
    return [Document(text=doc_text)]


@pytest.fixture
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
