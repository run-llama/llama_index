# Copyright Hewlett Packard Enterprise Development LP.

import pytest

from unittest.mock import MagicMock

from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.retrievers.alletra_x10000_retriever import AlletraX10000Retriever


def test_alletra_x10000_retriever_initialization():
    retriever = AlletraX10000Retriever(
        uri="http://example.com",
        s3_access_key="test_access_key",
        s3_secret_key="test_secret_key",
        collection_name="test_collection",
        search_config={"param1": "value1"},
        top_k=5,
    )
    assert retriever.uri == "http://example.com"
    assert retriever.access_key == "test_access_key"
    assert retriever.secret_key == "test_secret_key"
    assert retriever.collection_name == "test_collection"
    assert retriever.search_config == {"param1": "value1"}
    assert retriever.top_k == 5


def test_alletra_x10000_retriever_retrieve(mocker):
    mock_client = mocker.patch(
        "llama_index.retrievers.alletra_x10000_retriever.base.DIClient"
    )

    mock_response = MagicMock()
    mock_response = [
        {
            "dataChunk": "chunk1",
            "score": 0.9,
            "chunkMetadata": {
                "objectKey": "value",
                "startCharIndex": 1,
                "endCharIndex": 2,
                "bucketName": "string",
                "pageLabel": "string",
                "versionId": "string",
            },
        },
        {
            "dataChunk": "chunk2",
            "score": 0.8,
            "chunkMetadata": {
                "objectKey": "value",
                "startCharIndex": 1,
                "endCharIndex": 2,
                "bucketName": "string",
                "pageLabel": "string",
                "versionId": "string",
            },
        },
    ]
    mock_client.return_value.similarity_search.return_value = mock_response

    retriever = AlletraX10000Retriever(
        uri="http://example.com",
        s3_access_key="test_access_key",
        s3_secret_key="test_secret_key",
        collection_name="test_collection",
        search_config={"param1": "value1"},
        top_k=2,
    )
    query_bundle = QueryBundle(query_str="test query")
    result = retriever._retrieve(query_bundle)

    assert len(result) == 2
    assert isinstance(result[0], NodeWithScore)
    assert result[0].node.text == "chunk1"
    assert result[0].score == 0.9
    assert result[0].node.metadata["objectKey"] == "value"
    assert result[1].node.text == "chunk2"
    assert result[1].score == 0.8
    assert result[1].node.metadata["objectKey"] == "value"


def test_alletra_x10000_retriever_retrieve_empty_response(mocker):
    retriever = AlletraX10000Retriever(
        uri="http://example.com",
        s3_access_key="test_access_key",
        s3_secret_key="test_secret_key",
        collection_name="test_collection",
        search_config={"param1": "value1"},
        top_k=2,
    )

    # Mock the DIClient to return an empty response
    mock_client = mocker.patch(
        "llama_index.retrievers.alletra_x10000_retriever.base.DIClient"
    )
    mock_client.return_value.similarity_search.return_value.json.return_value = []

    # Call the method
    query_bundle = QueryBundle(query_str="test query")
    result = retriever._retrieve(query_bundle)

    # Assertions
    assert result == []


def test_alletra_x10000_retriever_retrieve_error_handling(mocker):
    mock_client = mocker.patch(
        "llama_index.retrievers.alletra_x10000_retriever.base.DIClient"
    )
    mock_client.return_value.similarity_search.side_effect = Exception("Test exception")

    retriever = AlletraX10000Retriever(
        uri="http://example.com",
        s3_access_key="test_access_key",
        s3_secret_key="test_secret_key",
        collection_name="test_collection",
        search_config={"param1": "value1"},
        top_k=2,
    )
    query_bundle = QueryBundle(query_str="test query")

    with pytest.raises(Exception, match="Test exception"):
        retriever._retrieve(query_bundle)
