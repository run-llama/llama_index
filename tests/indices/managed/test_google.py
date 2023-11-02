from unittest.mock import MagicMock, patch

import pytest
from llama_index.response.schema import PydanticResponse
from llama_index.schema import Document

try:
    import google.ai.generativelanguage as genai

    has_google = True
except ImportError:
    has_google = False

from llama_index.indices.managed.google.generativeai import GoogleIndex
from llama_index.response_synthesizers.google.generativeai import SynthesizedResponse

SKIP_TEST_REASON = "Google GenerativeAI is not installed"


if has_google:
    import llama_index.vector_stores.google.generativeai.genai_extension as genaix

    genaix.set_defaults(
        genaix.Config(api_endpoint="No-such-endpoint-to-prevent-hitting-real-backend")
    )


@pytest.mark.skipif(not has_google, reason=SKIP_TEST_REASON)
def test_from_corpus() -> None:
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
@patch("google.ai.generativelanguage.RetrieverServiceClient.create_chunk")
def test_from_documents(
    mock_create_chunk: MagicMock,
    mock_create_document: MagicMock,
    mock_create_corpus: MagicMock,
) -> None:
    def fake_create_corpus(request: genai.CreateCorpusRequest) -> genai.Corpus:
        return request.corpus

    # Arrange
    mock_create_corpus.side_effect = fake_create_corpus
    mock_create_document.return_value = genai.Document(
        name="corpora/123/documents/doc-456"
    )
    mock_create_chunk.return_value = genai.Chunk(
        name="corpora/123/documents/doc-456/chunks/789"
    )

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

    assert mock_create_chunk.call_count == 2
    create_chunk_requests = mock_create_chunk.call_args_list

    first_create_chunk_request = create_chunk_requests[0].args[0]
    assert first_create_chunk_request.chunk.data.string_value == "Hello, my darling"

    second_create_chunk_request = create_chunk_requests[1].args[0]
    assert second_create_chunk_request.chunk.data.string_value == "Goodbye, my baby"


@pytest.mark.skipif(not has_google, reason=SKIP_TEST_REASON)
@patch("google.ai.generativelanguage.RetrieverServiceClient.query_corpus")
@patch("google.ai.generativelanguage.TextServiceClient.generate_text_answer")
def test_as_query_engine(
    mock_generate_text_answer: MagicMock, mock_query_corpus: MagicMock
) -> None:
    # Arrange
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
    mock_generate_text_answer.return_value = genai.GenerateTextAnswerResponse(
        answer=genai.TextCompletion(
            output="42",
        ),
        attributed_passages=[
            genai.AttributedPassage(
                text="Meaning of life is 42.",
                passage_ids=["corpora/123/documents/456/chunks/789"],
            ),
        ],
        answerable_probability=0.8,
    )

    # Act
    index = GoogleIndex.from_corpus(corpus_id="123")
    query_engine = index.as_query_engine(answer_style=genai.AnswerStyle.EXTRACTIVE)
    response = query_engine.query("What is the meaning of life?")

    # Assert
    assert mock_query_corpus.call_count == 1
    query_corpus_request = mock_query_corpus.call_args.args[0]
    assert query_corpus_request.name == "corpora/123"
    assert query_corpus_request.query == "What is the meaning of life?"

    assert isinstance(response, PydanticResponse)
    reply = response.response
    assert isinstance(reply, SynthesizedResponse)

    assert reply.answer == "42"
    assert reply.attributed_passages == ["Meaning of life is 42."]
    assert reply.answerable_probability == pytest.approx(0.8)

    assert mock_generate_text_answer.call_count == 1
    generate_text_answer_request = mock_generate_text_answer.call_args.args[0]
    assert generate_text_answer_request.question.text == "What is the meaning of life?"
    assert generate_text_answer_request.answer_style == genai.AnswerStyle.EXTRACTIVE

    passages = generate_text_answer_request.grounding_source.passages.passages
    assert len(passages) == 1
    passage = passages[0]
    assert passage.text == "It's 42"
