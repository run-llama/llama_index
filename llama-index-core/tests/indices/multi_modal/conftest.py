"""Fixtures for MultiModalVectorStoreIndex tests."""

import pytest
from typing import Any, List, Sequence

from llama_index.core import MockEmbedding
from llama_index.core.embeddings import MockMultiModalEmbedding
from llama_index.core.schema import (
    BaseNode,
    Document,
    ImageDocument,
    ImageNode,
    TextNode,
)
from llama_index.core.vector_stores.simple import SimpleVectorStore


class MockVectorStoreWithSkipEmbedding(SimpleVectorStore):
    """Mock vector store that supports skip_embedding by handling None embeddings."""

    def add(
        self,
        nodes: Sequence[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """Add nodes to index, handling None embeddings."""
        from llama_index.core.vector_stores.utils import node_to_metadata_dict

        for node in nodes:
            embedding = node.embedding if node.embedding is not None else []
            self.data.embedding_dict[node.node_id] = embedding
            self.data.text_id_to_ref_doc_id[node.node_id] = node.ref_doc_id or "None"

            metadata = node_to_metadata_dict(
                node, remove_text=True, flat_metadata=False
            )
            metadata.pop("_node_content", None)
            self.data.metadata_dict[node.node_id] = metadata
        return [node.node_id for node in nodes]


@pytest.fixture()
def mock_text_embed_model():
    """Mock text embedding model with 5 dimensions."""
    return MockEmbedding(embed_dim=5)


@pytest.fixture()
def mock_image_embed_model():
    """Mock multimodal embedding model with 5 dimensions."""
    return MockMultiModalEmbedding(embed_dim=5)


@pytest.fixture()
def documents() -> List[Document]:
    """Sample documents for testing."""
    return [
        Document(text="Hello world.", id_="test_doc_1"),
        Document(text="This is a test.", id_="test_doc_2"),
    ]


@pytest.fixture()
def image_documents() -> List[ImageDocument]:
    """Sample image documents for testing."""
    base64_str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
    return [
        ImageDocument(
            image=base64_str, metadata={"file_name": "test1.png"}, id_="test_img_1"
        ),
        ImageDocument(
            image=base64_str, metadata={"file_name": "test2.png"}, id_="test_img_2"
        ),
    ]


@pytest.fixture()
def text_nodes() -> List[TextNode]:
    """Sample text nodes with pre-existing embeddings."""
    return [
        TextNode(
            text="Hello world.", embedding=[1.0, 0.0, 0.0, 0.0, 0.0], id_="node_1"
        ),
        TextNode(
            text="This is a test.", embedding=[0.0, 1.0, 0.0, 0.0, 0.0], id_="node_2"
        ),
    ]


@pytest.fixture()
def image_nodes() -> List[ImageNode]:
    """Sample image nodes with pre-existing embeddings."""
    base64_str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
    return [
        ImageNode(
            image=base64_str,
            embedding=[1.0, 0.0, 0.0, 0.0, 0.0],
            text_embedding=[0.0, 1.0, 0.0, 0.0, 0.0],
            id_="img_node_1",
        ),
        ImageNode(
            image=base64_str,
            embedding=[0.0, 0.0, 1.0, 0.0, 0.0],
            text_embedding=[0.0, 0.0, 1.0, 0.0, 0.0],
            id_="img_node_2",
        ),
    ]
