import json
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode
from llama_index.vector_stores.azureaisearch import (
    AzureAISearchVectorStore,
    IndexManagement,
)

try:
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.models import VectorizedQuery

    azureaisearch_installed = True
except ImportError:
    azureaisearch_installed = False
    search_client = None  # type: ignore


def create_mock_vector_store(
    search_client: Any,
    index_name: Optional[str] = None,
    index_management: IndexManagement = IndexManagement.NO_VALIDATION,
) -> AzureAISearchVectorStore:
    return AzureAISearchVectorStore(
        search_or_index_client=search_client,
        id_field_key="id",
        chunk_field_key="content",
        embedding_field_key="embedding",
        metadata_string_field_key="metadata",
        doc_id_field_key="doc_id",
        filterable_metadata_field_keys=[],  # Added to match the updated constructor
        hidden_field_keys=["embedding"],
        index_name=index_name,
        index_management=index_management,
        embedding_dimensionality=2,  # Assuming a dimensionality of 2 for simplicity
    )


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
    not azureaisearch_installed, reason="azure-search-documents package not installed"
)
def test_azureaisearch_add_two_batches() -> None:
    search_client = MagicMock(spec=SearchClient)

    with patch("azure.search.documents.IndexDocumentsBatch") as MockIndexDocumentsBatch:
        index_documents_batch_instance = MockIndexDocumentsBatch.return_value
        vector_store = create_mock_vector_store(search_client)

        nodes = create_sample_documents(11)
        ids = vector_store.add(nodes)

        call_count = index_documents_batch_instance.add_upload_actions.call_count

        assert ids is not None
        assert len(ids) == 11
        assert call_count == 11  # Adjust this value based on your logic
        assert search_client.index_documents.call_count == 1


@pytest.mark.skipif(
    not azureaisearch_installed, reason="azure-search-documents package not installed"
)
def test_azureaisearch_add_one_batch() -> None:
    search_client = MagicMock(spec=SearchClient)

    with patch("azure.search.documents.IndexDocumentsBatch") as MockIndexDocumentsBatch:
        index_documents_batch_instance = MockIndexDocumentsBatch.return_value
        vector_store = create_mock_vector_store(search_client)

        nodes = create_sample_documents(11)
        ids = vector_store.add(nodes)

        call_count = index_documents_batch_instance.add_upload_actions.call_count

        assert ids is not None
        assert len(ids) == 11
        assert call_count == 11  # Adjust this value based on your logic
        assert search_client.index_documents.call_count == 1


@pytest.mark.skipif(
    not azureaisearch_installed, reason="azure-search-documents package not installed"
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
    not azureaisearch_installed, reason="azure-search-documents package not installed"
)
def test_invalid_index_management_for_searchindexclient() -> None:
    search_client = MagicMock(spec=SearchIndexClient)

    # Index name must be supplied
    with pytest.raises(
        ValueError,
        match="index_name must be supplied if search_or_index_client is of type azure.search.documents.SearchIndexClient",
    ):
        create_mock_vector_store(
            search_client, index_management=IndexManagement.VALIDATE_INDEX
        )

    # No error when index name is supplied with SearchIndexClient
    create_mock_vector_store(
        search_client,
        index_name="test01",
        index_management=IndexManagement.CREATE_IF_NOT_EXISTS,
    )


@pytest.mark.skipif(
    not azureaisearch_installed, reason="azure-search-documents package not installed"
)
def test_azureaisearch_query() -> None:
    search_client = MagicMock(spec=SearchClient)

    # Mock the search method of the search client
    mock_search_results = [
        {
            "id": "test_id_1",
            "chunk": "test chunk 1",
            "content": "test chunk 1",
            "metadata": json.dumps({"key": "value1"}),
            "doc_id": "doc1",
            "@search.score": 0.9,
        },
        {
            "id": "test_id_2",
            "chunk": "test chunk 2",
            "content": "test chunk 2",
            "metadata": json.dumps({"key": "value2"}),
            "doc_id": "doc2",
            "@search.score": 0.8,
        },
    ]
    search_client.search.return_value = mock_search_results

    vector_store = create_mock_vector_store(search_client)

    # Create a sample query
    query = VectorStoreQuery(
        query_embedding=[0.1, 0.2],
        similarity_top_k=2,
        mode=VectorStoreQueryMode.DEFAULT,
    )

    # Execute the query
    result = vector_store.query(query)

    # Assert the search method was called with correct parameters
    search_client.search.assert_called_once_with(
        search_text="*",
        vector_queries=[
            VectorizedQuery(
                vector=[0.1, 0.2], k_nearest_neighbors=2, fields="embedding"
            )
        ],
        top=2,
        select=["id", "content", "metadata", "doc_id"],
        filter=None,
    )

    # Assert the result structure
    assert len(result.nodes) == 2
    assert len(result.ids) == 2
    assert len(result.similarities) == 2

    # Assert the content of the results
    assert result.nodes[0].text == "test chunk 1"
    assert result.nodes[1].text == "test chunk 2"
    assert result.ids == ["test_id_1", "test_id_2"]
    assert result.similarities == [0.9, 0.8]

    # Assert the metadata
    assert result.nodes[0].metadata == {"key": "value1"}
    assert result.nodes[1].metadata == {"key": "value2"}
