from unittest.mock import patch, MagicMock

from llama_index.core.schema import NodeWithScore, TextNode


@patch("llama_index.core.utilities.aws_utils.get_aws_service_client")
def test_retrieve(mock_get_aws_service_client):
    mock_client = MagicMock()
    mock_client.retrieve.return_value = {
        "retrievalResults": [
            {
                "content": {"text": "This is a test result."},
                "location": "test_location",
                "metadata": {
                    "x-amz-bedrock-kb-source-uri": "s3://bucket/fileName",
                    "key": "value",
                },
                "score": 0.8,
            },
            {
                "content": {"text": "Another test result."},
            },
        ]
    }
    mock_get_aws_service_client.return_value = mock_client
    knowledge_base_id = "test-knowledge-base-id"
    retrieval_config = {
        "vectorSearchConfiguration": {
            "numberOfResults": 2,
            "overrideSearchType": "SEMANTIC",
            "filter": {"equals": {"key": "tag", "value": "space"}},
        }
    }
    from llama_index.retrievers.bedrock import AmazonKnowledgeBasesRetriever

    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id=knowledge_base_id,
        retrieval_config=retrieval_config,
    )
    retriever._client = mock_client

    # Call the method being tested
    query = "Test query"
    result = retriever.retrieve(query)

    # Assert the expected output
    expected_result = [
        NodeWithScore(
            node=TextNode(
                text="This is a test result.",
                metadata={
                    "location": "test_location",
                    "sourceMetadata": {
                        "x-amz-bedrock-kb-source-uri": "s3://bucket/fileName",
                        "key": "value",
                    },
                },
            ),
            score=0.8,
        ),
        NodeWithScore(
            node=TextNode(text="Another test result.", metadata={}), score=0.0
        ),
    ]
    assert result[0].node.text == expected_result[0].node.text
    assert result[0].node.metadata == expected_result[0].node.metadata
    assert result[0].score == expected_result[0].score
    assert result[1].node.text == expected_result[1].node.text
    assert result[1].node.metadata == expected_result[1].node.metadata
    assert result[1].score == expected_result[1].score
