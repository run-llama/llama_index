from unittest.mock import patch, MagicMock

from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle


@patch("llama_index.core.utilities.aws_utils.get_aws_service_client")
def test_retrieve_document_results(mock_get_aws_service_client):
    mock_client = MagicMock()
    mock_client.query.return_value = {
        "ResultItems": [
            {
                "Type": "DOCUMENT",
                "DocumentId": "doc1",
                "DocumentTitle": {"Text": "Test Document"},
                "DocumentURI": "https://example.com/doc1",
                "DocumentExcerpt": {"Text": "This is a test result."},
                "ScoreAttributes": {"ScoreConfidence": "VERY_HIGH"},
            },
            {
                "Type": "DOCUMENT",
                "DocumentId": "doc2",
                "DocumentExcerpt": {"Text": "Another test result."},
                "ScoreAttributes": {"ScoreConfidence": "MEDIUM"},
            },
        ]
    }
    mock_get_aws_service_client.return_value = mock_client

    from llama_index.retrievers.kendra import AmazonKendraRetriever

    retriever = AmazonKendraRetriever(
        index_id="test-index-id", query_config={"PageSize": 2}
    )

    # Call the method being tested
    query = QueryBundle(query_str="Test query")
    result = retriever._retrieve(query)

    # Assert the expected output
    expected_result = [
        NodeWithScore(
            node=TextNode(
                text="This is a test result.",
                metadata={
                    "document_id": "doc1",
                    "title": "Test Document",
                    "source": "https://example.com/doc1",
                },
            ),
            score=1.0,  # VERY_HIGH maps to 1.0
        ),
        NodeWithScore(
            node=TextNode(
                text="Another test result.",
                metadata={"document_id": "doc2"},
            ),
            score=0.6,  # MEDIUM maps to 0.6
        ),
    ]

    assert len(result) == len(expected_result)
    for actual, expected in zip(result, expected_result):
        assert actual.node.text == expected.node.text
        assert actual.node.metadata == expected.node.metadata
        assert actual.score == expected.score

    # Verify Kendra API was called correctly
    mock_client.query.assert_called_once_with(
        IndexId="test-index-id", QueryText="Test query", PageSize=2
    )
