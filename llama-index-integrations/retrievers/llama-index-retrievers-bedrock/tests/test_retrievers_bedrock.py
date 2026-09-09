import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from llama_index.core.schema import QueryBundle


@patch("llama_index.core.utilities.aws_utils.get_aws_service_client")
@patch("aioboto3.Session")
def test_aretrieve_async(mock_aioboto3_session, mock_get_aws_service_client):
    """Test the async retrieve method."""
    # Setup sync client mock
    mock_sync_client = MagicMock()
    mock_get_aws_service_client.return_value = mock_sync_client

    # Setup async client mock
    mock_async_client = AsyncMock()
    mock_async_client.retrieve = AsyncMock(
        return_value={
            "retrievalResults": [
                {
                    "content": {"text": "Async test result 1."},
                    "location": "async_location_1",
                    "metadata": {
                        "x-amz-bedrock-kb-source-uri": "s3://bucket/file1.pdf",
                        "key": "value1",
                    },
                    "score": 0.9,
                },
                {
                    "content": {"text": "Async test result 2."},
                    "location": "async_location_2",
                    "metadata": {
                        "x-amz-bedrock-kb-source-uri": "s3://bucket/file2.pdf",
                    },
                    "score": 0.7,
                },
            ]
        }
    )

    # Setup async context manager for aioboto3 client
    mock_session = MagicMock()
    mock_session.client = MagicMock()
    mock_session.client.return_value.__aenter__ = AsyncMock(
        return_value=mock_async_client
    )
    mock_session.client.return_value.__aexit__ = AsyncMock(return_value=None)
    mock_aioboto3_session.return_value = mock_session

    from llama_index.retrievers.bedrock import AmazonKnowledgeBasesRetriever

    knowledge_base_id = "test-knowledge-base-id"
    retrieval_config = {
        "vectorSearchConfiguration": {
            "numberOfResults": 2,
            "overrideSearchType": "HYBRID",
        }
    }

    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id=knowledge_base_id,
        retrieval_config=retrieval_config,
    )

    # Run async test
    query_bundle = QueryBundle(query_str="Test async query")
    result = asyncio.run(retriever._aretrieve(query_bundle))

    # Assertions
    assert len(result) == 2
    assert result[0].node.text == "Async test result 1."
    assert result[0].score == 0.9
    assert result[0].node.metadata["location"] == "async_location_1"
    assert (
        result[0].node.metadata["sourceMetadata"]["x-amz-bedrock-kb-source-uri"]
        == "s3://bucket/file1.pdf"
    )

    assert result[1].node.text == "Async test result 2."
    assert result[1].score == 0.7
    assert result[1].node.metadata["location"] == "async_location_2"


@patch("llama_index.core.utilities.aws_utils.get_aws_service_client")
@patch("aioboto3.Session")
def test_aretrieve_with_missing_fields(
    mock_aioboto3_session, mock_get_aws_service_client
):
    """Test async retrieve method handles missing optional fields gracefully."""
    mock_sync_client = MagicMock()
    mock_get_aws_service_client.return_value = mock_sync_client

    # Response with minimal fields (no location, metadata, or score)
    mock_async_client = AsyncMock()
    mock_async_client.retrieve = AsyncMock(
        return_value={
            "retrievalResults": [
                {
                    "content": {"text": "Minimal result."},
                },
            ]
        }
    )

    mock_session = MagicMock()
    mock_session.client = MagicMock()
    mock_session.client.return_value.__aenter__ = AsyncMock(
        return_value=mock_async_client
    )
    mock_session.client.return_value.__aexit__ = AsyncMock(return_value=None)
    mock_aioboto3_session.return_value = mock_session

    from llama_index.retrievers.bedrock import AmazonKnowledgeBasesRetriever

    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id="test-kb-id",
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 1}},
    )

    query_bundle = QueryBundle(query_str="Minimal query")
    result = asyncio.run(retriever._aretrieve(query_bundle))

    # Should handle missing fields gracefully
    assert len(result) == 1
    assert result[0].node.text == "Minimal result."
    assert result[0].node.metadata == {}
    assert result[0].score == 0.0


@patch("llama_index.core.utilities.aws_utils.get_aws_service_client")
def test_retrieve_sync_still_works(mock_get_aws_service_client):
    """Test that the original sync _retrieve method still works (backward compatibility)."""
    mock_client = MagicMock()
    mock_client.retrieve.return_value = {
        "retrievalResults": [
            {
                "content": {"text": "Sync test result."},
                "location": "sync_location",
                "metadata": {"key": "value"},
                "score": 0.85,
            },
        ]
    }
    mock_get_aws_service_client.return_value = mock_client

    from llama_index.retrievers.bedrock import AmazonKnowledgeBasesRetriever

    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id="test-kb-id",
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 1}},
    )
    retriever._client = mock_client

    query_bundle = QueryBundle(query_str="Sync query")
    result = retriever._retrieve(query_bundle)

    # Verify sync method still works
    assert len(result) == 1
    assert result[0].node.text == "Sync test result."
    assert result[0].score == 0.85
    assert result[0].node.metadata["location"] == "sync_location"


@patch("llama_index.core.utilities.aws_utils.get_aws_service_client")
@patch("aioboto3.Session")
def test_aretrieve_concurrent_calls(mock_aioboto3_session, mock_get_aws_service_client):
    """Test multiple concurrent async retrieve calls."""
    mock_sync_client = MagicMock()
    mock_get_aws_service_client.return_value = mock_sync_client

    # Create different responses for different queries
    async def mock_retrieve_side_effect(**kwargs):
        query_text = kwargs.get("retrievalQuery", {}).get("text", "")
        return {
            "retrievalResults": [
                {
                    "content": {"text": f"Result for: {query_text}"},
                    "score": 0.8,
                },
            ]
        }

    mock_async_client = AsyncMock()
    mock_async_client.retrieve = AsyncMock(side_effect=mock_retrieve_side_effect)

    mock_session = MagicMock()
    mock_session.client = MagicMock()
    mock_session.client.return_value.__aenter__ = AsyncMock(
        return_value=mock_async_client
    )
    mock_session.client.return_value.__aexit__ = AsyncMock(return_value=None)
    mock_aioboto3_session.return_value = mock_session

    from llama_index.retrievers.bedrock import AmazonKnowledgeBasesRetriever

    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id="test-kb-id",
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 1}},
    )

    # Run concurrent async calls
    async def run_concurrent_queries():
        tasks = [
            retriever._aretrieve(QueryBundle(query_str=f"Query {i}")) for i in range(3)
        ]
        return await asyncio.gather(*tasks)

    results = asyncio.run(run_concurrent_queries())

    # Verify all concurrent calls completed
    assert len(results) == 3
    assert all(len(r) == 1 for r in results)
    assert "Query 0" in results[0][0].node.text
    assert "Query 1" in results[1][0].node.text
    assert "Query 2" in results[2][0].node.text


@patch("llama_index.core.utilities.aws_utils.get_aws_service_client")
def test_managed_kb_retrieve(mock_get_aws_service_client):
    """Test that passing managed_search_config results in the correct API call with managedSearchConfiguration."""
    mock_client = MagicMock()
    mock_client.retrieve.return_value = {
        "retrievalResults": [
            {
                "content": {"text": "Managed KB result."},
                "location": "managed_location",
                "score": 0.92,
            },
        ]
    }
    mock_get_aws_service_client.return_value = mock_client

    from llama_index.retrievers.bedrock import AmazonKnowledgeBasesRetriever

    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id="managed-kb-id",
        managed_search_config={"numberOfResults": 5},
    )
    retriever._client = mock_client

    query_bundle = QueryBundle(query_str="Managed query")
    result = retriever._retrieve(query_bundle)

    # Verify the API was called with managedSearchConfiguration
    mock_client.retrieve.assert_called_once_with(
        retrievalQuery={"text": "Managed query"},
        knowledgeBaseId="managed-kb-id",
        retrievalConfiguration={"managedSearchConfiguration": {"numberOfResults": 5}},
    )

    # Verify results are parsed correctly
    assert len(result) == 1
    assert result[0].node.text == "Managed KB result."
    assert result[0].score == 0.92


@patch("llama_index.core.utilities.aws_utils.get_aws_service_client")
def test_managed_kb_type_auto_config(mock_get_aws_service_client):
    """Test that knowledge_base_type='MANAGED' auto-uses managed config."""
    mock_client = MagicMock()
    mock_client.retrieve.return_value = {
        "retrievalResults": [
            {
                "content": {"text": "Auto managed result."},
                "score": 0.88,
            },
        ]
    }
    mock_get_aws_service_client.return_value = mock_client

    from llama_index.retrievers.bedrock import AmazonKnowledgeBasesRetriever

    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id="auto-managed-kb-id",
        knowledge_base_type="MANAGED",
    )
    retriever._client = mock_client

    query_bundle = QueryBundle(query_str="Auto managed query")
    result = retriever._retrieve(query_bundle)

    # Verify the API was called with an empty managedSearchConfiguration
    mock_client.retrieve.assert_called_once_with(
        retrievalQuery={"text": "Auto managed query"},
        knowledgeBaseId="auto-managed-kb-id",
        retrievalConfiguration={"managedSearchConfiguration": {}},
    )

    assert len(result) == 1
    assert result[0].node.text == "Auto managed result."
    assert result[0].score == 0.88


@patch("llama_index.core.utilities.aws_utils.get_aws_service_client")
def test_vector_kb_backward_compat(mock_get_aws_service_client):
    """Test that existing vector config still works unchanged (backward compatibility)."""
    mock_client = MagicMock()
    mock_client.retrieve.return_value = {
        "retrievalResults": [
            {
                "content": {"text": "Vector result."},
                "location": "vector_location",
                "metadata": {"key": "value"},
                "score": 0.95,
            },
        ]
    }
    mock_get_aws_service_client.return_value = mock_client

    from llama_index.retrievers.bedrock import AmazonKnowledgeBasesRetriever

    vector_config = {
        "vectorSearchConfiguration": {
            "numberOfResults": 4,
            "overrideSearchType": "SEMANTIC",
            "filter": {"equals": {"key": "tag", "value": "space"}},
        }
    }

    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id="vector-kb-id",
        retrieval_config=vector_config,
    )
    retriever._client = mock_client

    query_bundle = QueryBundle(query_str="Vector query")
    result = retriever._retrieve(query_bundle)

    # Verify the API was called with the original vectorSearchConfiguration
    mock_client.retrieve.assert_called_once_with(
        retrievalQuery={"text": "Vector query"},
        knowledgeBaseId="vector-kb-id",
        retrievalConfiguration=vector_config,
    )

    # Verify results
    assert len(result) == 1
    assert result[0].node.text == "Vector result."
    assert result[0].score == 0.95
    assert result[0].node.metadata["location"] == "vector_location"
    assert result[0].node.metadata["sourceMetadata"] == {"key": "value"}


@patch("boto3.Session")
def test_agentic_retrieve_helper(mock_session):
    """Test agentic_retrieve() standalone helper."""
    from llama_index.retrievers.bedrock import agentic_retrieve

    mock_client = MagicMock()
    mock_session.return_value.client.return_value = mock_client
    mock_client.agentic_retrieve_stream.return_value = {
        "stream": [
            {
                "result": {
                    "results": [
                        {"content": {"text": "agentic llama result"}, "score": 0.88}
                    ]
                }
            }
        ]
    }

    result = agentic_retrieve(
        knowledge_base_id="test-kb",
        query="test query",
        region_name="us-west-2",
    )

    assert "results" in result
    assert len(result["results"]) == 1
    assert result["results"][0]["content"]["text"] == "agentic llama result"
    mock_client.agentic_retrieve_stream.assert_called_once()


@patch("boto3.Session")
def test_agentic_retrieve_with_generate_response(mock_session):
    """Test agentic_retrieve() with generate_response=True returns cited answer."""
    from llama_index.retrievers.bedrock import agentic_retrieve

    mock_client = MagicMock()
    mock_session.return_value.client.return_value = mock_client
    mock_client.agentic_retrieve_stream.return_value = {
        "stream": [
            {
                "result": {
                    "results": [{"content": {"text": "source chunk 1"}, "score": 0.9}],
                    "generatedResponse": {
                        "answer": "Based on the documents, the answer is X.",
                        "citations": [
                            {
                                "generatedResponsePart": {
                                    "textResponsePart": {
                                        "span": {"start": 0, "end": 40}
                                    }
                                }
                            }
                        ],
                    },
                }
            },
        ]
    }

    result = agentic_retrieve(
        knowledge_base_id="test-kb",
        query="What is X?",
        region_name="us-west-2",
        generate_response=True,
    )

    # Verify generateResponse was passed to the API
    call_kwargs = mock_client.agentic_retrieve_stream.call_args[1]
    assert call_kwargs["generateResponse"] is True

    # Verify generated answer is returned
    assert (
        result["generatedResponse"]["answer"]
        == "Based on the documents, the answer is X."
    )
    assert len(result["generatedResponse"]["citations"]) == 1

    # Verify results are also present
    assert len(result["results"]) == 1
    assert result["results"][0]["content"]["text"] == "source chunk 1"
