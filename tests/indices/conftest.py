from typing import List
import pytest
from llama_index.data_structs.node import DocumentRelationship, Node

from llama_index.readers.schema.base import Document


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
    return [Document(doc_text)]


@pytest.fixture
def nodes() -> List[Node]:
    """Get documents."""
    # NOTE: one document for now
    return [
        Node("Hello world.", relationships={DocumentRelationship.SOURCE: "test doc"}),
        Node(
            "This is a test.", relationships={DocumentRelationship.SOURCE: "test doc"}
        ),
        Node(
            "This is another test.",
            relationships={DocumentRelationship.SOURCE: "test doc"},
        ),
        Node(
            "This is a test v2.",
            relationships={DocumentRelationship.SOURCE: "test doc"},
        ),
    ]
