from unittest.mock import MagicMock, patch

import pytest
from llama_index.legacy.core.response.schema import Response
from llama_index.legacy.schema import Document

try:
    import google.ai.generativelanguage as genai

    has_google = True
except ImportError:
    has_google = False

from llama_index.legacy.indices.managed.google.generativeai import (
    GoogleIndex,
    set_google_config,
)

SKIP_TEST_REASON = "Google GenerativeAI is not installed"


if has_google:
    import llama_index.legacy.vector_stores.google.generativeai.genai_extension as genaix

    set_google_config(
        api_endpoint="No-such-endpoint-to-prevent-hitting-real-backend",
        testing=True,
    )


@pytest.mark.skipif(not has_google, reason=SKIP_TEST_REASON)
@patch("google.auth.credentials.Credentials")
def test_set_google_config(mock_credentials: MagicMock) -> None:
    set_google_config(auth_credentials=mock_credentials)
    config = genaix.get_config()
    assert config.auth_credentials == mock_credentials


@pytest.mark.skipif(not has_google, reason=SKIP_TEST_REASON)
@patch("google.ai.generativelanguage.RetrieverServiceClient.get_corpus")
def test_from_corpus(mock_get_corpus: MagicMock) -> None:
    # Arrange
    mock_get_corpus.return_value = genai.Corpus(name="corpora/123")

    # Act
    store = GoogleIndex.from_corpus(corpus_id="123")

    # Assert
    assert store.corpus_id == "123"


@pytest.mark.skipif(not has_google, reason=SKIP_TEST_REASON)
@patch("google.ai.generativelanguage.RetrieverServiceClient.create_corpus")
def test_create_corpus(mock_create_corpus: MagicMock) -> None:
    def fake_create_corpus(request: genai.CreateCorpusRequest) -> genai.Corpus:
        return request.corpus

    # Arrange
    mock_create_corpus.side_effect = fake_create_corpus

    # Act
    store = GoogleIndex.create_corpus(display_name="My first corpus")

    # Assert
    assert len(store.corpus_id) > 0
    assert mock_create_corpus.call_count == 1

    request = mock_create_corpus.call_args.args[0]
    assert request.corpus.name == f"corpora/{store.corpus_id}"
    assert request.corpus.display_name == "My first corpus"


@pytest.mark.skipif(not has_google, reason=SKIP_TEST_REASON)
@patch("google.ai.generativelanguage.RetrieverServiceClient.create_corpus")
@patch("google.ai.generativelanguage.RetrieverServiceClient.create_document")
@patch("google.ai.generativelanguage.RetrieverServiceClient.batch_create_chunks")
@patch("google.ai.generativelanguage.RetrieverServiceClient.get_document")
def test_from_documents(
    mock_get_document: MagicMock,
    mock_batch_create_chunk: MagicMock,
    mock_create_document: MagicMock,
    mock_create_corpus: MagicMock,
) -> None:
    from google.api_core import exceptions as gapi_exception

    def fake_create_corpus(request: genai.CreateCorpusRequest) -> genai.Corpus:
        return request.corpus

    # Arrange
    mock_get_document.side_effect = gapi_exception.NotFound("")
    mock_create_corpus.side_effect = fake_create_corpus
    mock_create_document.return_value = genai.Document(name="corpora/123/documents/456")
    mock_batch_create_chunk.side_effect = [
        genai.BatchCreateChunksResponse(
            chunks=[
                genai.Chunk(name="corpora/123/documents/456/chunks/777"),
            ]
        ),
        genai.BatchCreateChunksResponse(
            chunks=[
                genai.Chunk(name="corpora/123/documents/456/chunks/888"),
            ]
        ),
    ]

    # Act
    index = GoogleIndex.from_documents(
        [
            Document(text="Hello, my darling"),
            Document(text="Goodbye, my baby"),
        ]
    )

    # Assert
    assert mock_create_corpus.call_count == 1
    create_corpus_request = mock_create_corpus.call_args.args[0]
    assert create_corpus_request.corpus.name == f"corpora/{index.corpus_id}"

    create_document_request = mock_create_document.call_args.args[0]
    assert create_document_request.parent == f"corpora/{index.corpus_id}"

    assert mock_batch_create_chunk.call_count == 2

    first_batch_request = mock_batch_create_chunk.call_args_list[0].args[0]
    assert (
        first_batch_request.requests[0].chunk.data.string_value == "Hello, my darling"
    )

    second_batch_request = mock_batch_create_chunk.call_args_list[1].args[0]
    assert (
        second_batch_request.requests[0].chunk.data.string_value == "Goodbye, my baby"
    )


@pytest.mark.skipif(not has_google, reason=SKIP_TEST_REASON)
@patch("google.ai.generativelanguage.RetrieverServiceClient.query_corpus")
@patch("google.ai.generativelanguage.GenerativeServiceClient.generate_answer")
@patch("google.ai.generativelanguage.RetrieverServiceClient.get_corpus")
def test_as_query_engine(
    mock_get_corpus: MagicMock,
    mock_generate_answer: MagicMock,
    mock_query_corpus: MagicMock,
) -> None:
    # Arrange
    mock_get_corpus.return_value = genai.Corpus(name="corpora/123")
    mock_query_corpus.return_value = genai.QueryCorpusResponse(
        relevant_chunks=[
            genai.RelevantChunk(
                chunk=genai.Chunk(
                    name="corpora/123/documents/456/chunks/789",
                    data=genai.ChunkData(string_value="It's 42"),
                ),
                chunk_relevance_score=0.9,
            )
        ]
    )
    mock_generate_answer.return_value = genai.GenerateAnswerResponse(
        answer=genai.Candidate(
            content=genai.Content(parts=[genai.Part(text="42")]),
            grounding_attributions=[
                genai.GroundingAttribution(
                    content=genai.Content(
                        parts=[genai.Part(text="Meaning of life is 42")]
                    ),
                    source_id=genai.AttributionSourceId(
                        grounding_passage=genai.AttributionSourceId.GroundingPassageId(
                            passage_id="corpora/123/documents/456/chunks/777",
                            part_index=0,
                        )
                    ),
                ),
                genai.GroundingAttribution(
                    content=genai.Content(parts=[genai.Part(text="Or maybe not")]),
                    source_id=genai.AttributionSourceId(
                        grounding_passage=genai.AttributionSourceId.GroundingPassageId(
                            passage_id="corpora/123/documents/456/chunks/888",
                            part_index=0,
                        )
                    ),
                ),
            ],
            finish_reason=genai.Candidate.FinishReason.STOP,
        ),
        answerable_probability=0.9,
    )

    # Act
    index = GoogleIndex.from_corpus(corpus_id="123")
    query_engine = index.as_query_engine(
        answer_style=genai.GenerateAnswerRequest.AnswerStyle.EXTRACTIVE
    )
    response = query_engine.query("What is the meaning of life?")

    # Assert
    assert mock_query_corpus.call_count == 1
    query_corpus_request = mock_query_corpus.call_args.args[0]
    assert query_corpus_request.name == "corpora/123"
    assert query_corpus_request.query == "What is the meaning of life?"

    assert isinstance(response, Response)

    assert response.response == "42"

    assert mock_generate_answer.call_count == 1
    generate_answer_request = mock_generate_answer.call_args.args[0]
    assert (
        generate_answer_request.contents[0].parts[0].text
        == "What is the meaning of life?"
    )
    assert (
        generate_answer_request.answer_style
        == genai.GenerateAnswerRequest.AnswerStyle.EXTRACTIVE
    )

    passages = generate_answer_request.inline_passages.passages
    assert len(passages) == 1
    passage = passages[0]
    assert passage.content.parts[0].text == "It's 42"
