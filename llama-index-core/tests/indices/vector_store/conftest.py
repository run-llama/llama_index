"""Fixtures for VectorStore tests."""

import pytest
from typing import Any, List, Sequence

from llama_index.core.schema import BaseNode, Document
from tests.indices.multi_modal.conftest import MockVectorStoreWithSkipEmbedding


class MockVectorStoreGeneratesEmbeddings(MockVectorStoreWithSkipEmbedding):
    """Mock vector store that claims to generate embeddings natively."""

    @property
    def generates_embeddings(self) -> bool:
        """This vector store generates embeddings server-side."""
        return True

    def add(
        self,
        nodes: Sequence[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """Add nodes, handling native embedding generation."""
        from llama_index.core.vector_stores.utils import node_to_metadata_dict

        for node in nodes:
            # When generates_embeddings=True, embeddings should be None
            # The vector store is responsible for generating them
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
def documents() -> List[Document]:
    """Sample documents for testing - used by both skip_embedding and other tests."""
    return [
        Document(text="Hello world.", id_="doc_1"),
        Document(text="This is a test.", id_="doc_2"),
        Document(text="This is another test.", id_="doc_3"),
        Document(text="This is a test v2.", id_="doc_4"),
    ]
