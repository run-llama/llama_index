from unittest.mock import MagicMock
from typing import Any, List, Optional

from llama_index.schema import NodeRelationship, RelatedNodeInfo, TextNode

from llama_index.vector_stores import CognitiveSearchVectorStore
from llama_index.vector_stores.cogsearch import IndexManagement
import pytest

try:
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient

    cogsearch_installed = True
except ImportError:
    cogsearch_installed = False
    search_client = None  # type: ignore


def create_mock_vector_store(
    search_client: Any,
    index_name: Optional[str] = None,
    index_management: IndexManagement = IndexManagement.NO_VALIDATION,
) -> CognitiveSearchVectorStore:
    vector_store = CognitiveSearchVectorStore(
        search_or_index_client=search_client,
        id_field_key="id",
        chunk_field_key="content",
        embedding_field_key="embedding",
        metadata_string_field_key="li_jsonMetadata",
        doc_id_field_key="li_doc_id",
        index_name=index_name,
        index_management=index_management,
    )
    return vector_store


def create_sample_documents(n: int) -> List[TextNode]:
    nodes: List[TextNode] = []

    for i in range(n):
        nodes.append(
            TextNode(
                text=f"test node text {i}",
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(node_id=f"test doc id {i}")
                },
                embedding=[0.5, 0.5],
            )
        )

    return nodes


@pytest.mark.skipif(
    not cogsearch_installed, reason="azure-search-documents package not installed"
)
def test_cogsearch_add_two_batches() -> None:
    search_client = MagicMock(spec=SearchClient)
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
    search_client = MagicMock(spec=SearchClient)
    vector_store = create_mock_vector_store(search_client)

    nodes = create_sample_documents(10)

    ids = vector_store.add(nodes)

    call_count = search_client.merge_or_upload_documents.call_count

    assert ids is not None
    assert len(ids) == 10
    assert call_count == 1


@pytest.mark.skipif(
    not cogsearch_installed, reason="azure-search-documents package not installed"
)
def test_invalid_index_management_for_searchclient() -> None:
    search_client = MagicMock(spec=SearchClient)

    # No error
    create_mock_vector_store(
        search_client, index_management=IndexManagement.VALIDATE_INDEX
    )

    # Cannot supply index name
    # ruff: noqa: E501
    with pytest.raises(
        ValueError,
        match="index_name cannot be supplied if search_or_index_client is of type azure.search.documents.SearchClient",
    ):
        create_mock_vector_store(search_client, index_name="test01")

    # SearchClient cannot create an index
    with pytest.raises(ValueError):
        create_mock_vector_store(
            search_client,
            index_management=IndexManagement.CREATE_IF_NOT_EXISTS,
        )


@pytest.mark.skipif(
    not cogsearch_installed, reason="azure-search-documents package not installed"
)
def test_invalid_index_management_for_searchindexclient() -> None:
    search_client = MagicMock(spec=SearchIndexClient)

    # Index name must be supplied
    with pytest.raises(ValueError):
        create_mock_vector_store(
            search_client, index_management=IndexManagement.VALIDATE_INDEX
        )
