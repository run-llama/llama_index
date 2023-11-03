from unittest.mock import MagicMock, patch

import pytest
from llama_index.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.vector_stores.types import (
    ExactMatchFilter,
    MetadataFilters,
    VectorStoreQuery,
)

try:
    import google.ai.generativelanguage as genai

    has_google = True
except ImportError:
    has_google = False

from llama_index.vector_stores.google.generativeai import (
    GoogleVectorStore,
)

SKIP_TEST_REASON = "Google GenerativeAI is not installed"


if has_google:
    import llama_index.vector_stores.google.generativeai.genai_extension as genaix

    # Make sure the tests do not hit actual production servers.
    genaix.set_defaults(
        genaix.Config(api_endpoint="No-such-endpoint-to-prevent-hitting-real-backend")
    )


@pytest.mark.skipif(not has_google, reason=SKIP_TEST_REASON)
@patch("google.ai.generativelanguage.RetrieverServiceClient.create_corpus")
def test_create_corpus(mock_create_corpus: MagicMock) -> None:
    def fake_create_corpus(request: genai.CreateCorpusRequest) -> genai.Corpus:
        return request.corpus

    # Arrange
    mock_create_corpus.side_effect = fake_create_corpus

    # Act
    store = GoogleVectorStore.create_corpus(display_name="My first corpus")

    # Assert
    assert len(store.corpus_id) > 0
    assert mock_create_corpus.call_count == 1

    request = mock_create_corpus.call_args.args[0]
    assert request.corpus.name == f"corpora/{store.corpus_id}"
    assert request.corpus.display_name == "My first corpus"


@pytest.mark.skipif(not has_google, reason=SKIP_TEST_REASON)
@patch("google.ai.generativelanguage.RetrieverServiceClient.get_corpus")
def test_from_corpus(mock_get_corpus: MagicMock) -> None:
    # Arrange
    mock_get_corpus.return_value = genai.Corpus(name="corpora/123")

    # Act
    store = GoogleVectorStore.from_corpus(corpus_id="123")

    # Assert
    assert store.corpus_id == "123"


@pytest.mark.skipif(not has_google, reason=SKIP_TEST_REASON)
def test_class_name() -> None:
    # Act
    class_name = GoogleVectorStore.class_name()

    # Assert
    assert class_name == "GoogleVectorStore"


@pytest.mark.skipif(not has_google, reason=SKIP_TEST_REASON)
@patch("google.ai.generativelanguage.RetrieverServiceClient.create_chunk")
@patch("google.ai.generativelanguage.RetrieverServiceClient.create_document")
@patch("google.ai.generativelanguage.RetrieverServiceClient.get_corpus")
def test_add(
    mock_get_corpus: MagicMock,
    mock_create_document: MagicMock,
    mock_create_chunk: MagicMock,
) -> None:
    # Arrange
    mock_get_corpus.return_value = genai.Corpus(name="corpora/123")
    mock_create_document.return_value = genai.Document(
        name="corpora/123/documents/doc-456"
    )
    mock_create_chunk.return_value = genai.Chunk(
        name="corpora/123/documents/doc-456/chunks/789"
    )

    # Act
    store = GoogleVectorStore.from_corpus(corpus_id="123")
    store.add(
        [
            TextNode(
                text="Hello, my darling",
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(
                        node_id="doc-456",
                        metadata={"file_name": "Title for doc-456"},
                    )
                },
                metadata={"position": 100},
            ),
            TextNode(
                text="Goodbye, my baby",
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(
                        node_id="doc-456",
                        metadata={"file_name": "Title for doc-456"},
                    )
                },
                metadata={"position": 200},
            ),
        ]
    )

    # Assert
    create_document_request = mock_create_document.call_args.args[0]
    assert create_document_request == genai.CreateDocumentRequest(
        parent="corpora/123",
        document=genai.Document(
            name="corpora/123/documents/doc-456",
            display_name="Title for doc-456",
            custom_metadata=[
                genai.CustomMetadata(
                    key="file_name",
                    string_value="Title for doc-456",
                ),
            ],
        ),
    )

    assert mock_create_chunk.call_count == 2
    create_chunk_requests = mock_create_chunk.call_args_list

    first_create_chunk_request = create_chunk_requests[0].args[0]
    assert first_create_chunk_request == genai.CreateChunkRequest(
        parent="corpora/123/documents/doc-456",
        chunk=genai.Chunk(
            data=genai.ChunkData(string_value="Hello, my darling"),
            custom_metadata=[
                genai.CustomMetadata(
                    key="position",
                    numeric_value=100,
                ),
            ],
        ),
    )

    second_create_chunk_request = create_chunk_requests[1].args[0]
    assert second_create_chunk_request == genai.CreateChunkRequest(
        parent="corpora/123/documents/doc-456",
        chunk=genai.Chunk(
            data=genai.ChunkData(string_value="Goodbye, my baby"),
            custom_metadata=[
                genai.CustomMetadata(
                    key="position",
                    numeric_value=200,
                ),
            ],
        ),
    )


@pytest.mark.skipif(not has_google, reason=SKIP_TEST_REASON)
@patch("google.ai.generativelanguage.RetrieverServiceClient.delete_document")
@patch("google.ai.generativelanguage.RetrieverServiceClient.get_corpus")
def test_delete(mock_get_corpus: MagicMock, mock_delete_document: MagicMock) -> None:
    # Arrange
    mock_get_corpus.return_value = genai.Corpus(name="corpora/123")

    # Act
    store = GoogleVectorStore.from_corpus(corpus_id="123")
    store.delete(ref_doc_id="doc-456")

    # Assert
    delete_document_request = mock_delete_document.call_args.args[0]
    assert delete_document_request == genai.DeleteDocumentRequest(
        name="corpora/123/documents/doc-456",
        force=True,
    )


@pytest.mark.skipif(not has_google, reason=SKIP_TEST_REASON)
@patch("google.ai.generativelanguage.RetrieverServiceClient.query_corpus")
@patch("google.ai.generativelanguage.RetrieverServiceClient.get_corpus")
def test_query(mock_get_corpus: MagicMock, mock_query_corpus: MagicMock) -> None:
    # Arrange
    mock_get_corpus.return_value = genai.Corpus(name="corpora/123")
    mock_query_corpus.return_value = genai.QueryCorpusResponse(
        relevant_chunks=[
            genai.RelevantChunk(
                chunk=genai.Chunk(
                    name="corpora/123/documents/456/chunks/789",
                    data=genai.ChunkData(string_value="42"),
                ),
                chunk_relevance_score=0.9,
            )
        ]
    )

    # Act
    store = GoogleVectorStore.from_corpus(corpus_id="123")
    store.query(
        query=VectorStoreQuery(
            query_str="What is the meaning of life?",
            filters=MetadataFilters(
                filters=[
                    ExactMatchFilter(
                        key="author",
                        value="Arthur Schopenhauer",
                    )
                ]
            ),
            similarity_top_k=1,
        )
    )

    # Assert
    assert mock_query_corpus.call_count == 1
    query_corpus_request = mock_query_corpus.call_args.args[0]
    assert query_corpus_request == genai.QueryCorpusRequest(
        name="corpora/123",
        query="What is the meaning of life?",
        metadata_filters=[
            genai.MetadataFilter(
                key="author",
                conditions=[
                    genai.Condition(
                        operation=genai.Condition.Operator.EQUAL,
                        string_value="Arthur Schopenhauer",
                    )
                ],
            )
        ],
        results_count=1,
    )
