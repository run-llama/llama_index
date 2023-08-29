import sys
from unittest.mock import MagicMock
from typing import Any, List

from llama_index.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.vector_stores.types import NodeWithEmbedding

from llama_index.vector_stores import CognitiveSearchVectorStore
import pytest

try:
    from azure.search.documents import SearchClient
except ImportError:
    search_client = None  # type: ignore


try:
    import azure.search.documents

    cogsearch_installed = True
except ImportError:
    cogsearch_installed = False


def create_mock_vector_store(search_client: Any) -> CognitiveSearchVectorStore:
    vector_store = CognitiveSearchVectorStore(
        search_client,
        id_field_key="id",
        chunk_field_key="content",
        embedding_field_key="embedding",
        metadata_field_key="li_jsonMetadata",
        doc_id_field_key="li_doc_id",
    )
    return vector_store


def create_sample_documents(n: int) -> List[NodeWithEmbedding]:
    nodes: List[NodeWithEmbedding] = []

    for i in range(n):
        nodes.append(
            NodeWithEmbedding(
                node=TextNode(
                    text=f"test node text {i}",
                    relationships={
                        NodeRelationship.SOURCE: RelatedNodeInfo(
                            node_id=f"test doc id {i}"
                        )
                    },
                ),
                embedding=[0.5, 0.5],
            )
        )

    return nodes


@pytest.mark.skipif(
    not cogsearch_installed, reason="azure-search-documents package not installed"
)
def test_cogsearch_add_two_batches() -> None:
    search_client = MagicMock()
    vector_store = create_mock_vector_store(search_client)

    nodes = create_sample_documents(11)

    ids = vector_store.add(nodes)

    call_count = search_client.merge_or_upload_documents.call_count

    assert ids is not None
    assert len(ids) == 11
    assert call_count == 2


@pytest.mark.skipif(
    not cogsearch_installed, reason="azure-search-documents package not installed"
)
def test_cogsearch_add_one_batch() -> None:
    search_client = MagicMock()
    vector_store = create_mock_vector_store(search_client)

    nodes = create_sample_documents(10)

    ids = vector_store.add(nodes)

    call_count = search_client.merge_or_upload_documents.call_count

    assert ids is not None
    assert len(ids) == 10
    assert call_count == 1
