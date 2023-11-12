from unittest.mock import MagicMock, patch

import pytest

try:
    import google.ai.generativelanguage as genai

    has_google = True
except ImportError:
    has_google = False

from llama_index.response_synthesizers.google.generativeai import (
    GoogleTextSynthesizer,
)
from llama_index.schema import NodeWithScore, TextNode

SKIP_TEST_REASON = "Google GenerativeAI is not installed"


if has_google:
    import llama_index.vector_stores.google.generativeai.genai_extension as genaix

    genaix.set_defaults(
        genaix.Config(api_endpoint="No-such-endpoint-to-prevent-hitting-real-backend")
    )


@pytest.mark.skipif(not has_google, reason=SKIP_TEST_REASON)
@patch("google.ai.generativelanguage.TextServiceClient.generate_text_answer")
def test_get_response(mock_generate_text_answer: MagicMock) -> None:
    # Arrange
    mock_generate_text_answer.return_value = genai.GenerateTextAnswerResponse(
        answer=genai.TextCompletion(
            output="42",
            citation_metadata=genai.CitationMetadata(
                citation_sources=[
                    genai.CitationSource(
                        start_index=100,
                        end_index=200,
                        uri="answer.com/meaning_of_life.txt",
                    )
                ]
            ),
        ),
        attributed_passages=[
            genai.AttributedPassage(
                text="Meaning of life is 42.",
                passage_ids=["corpora/123/documents/456/chunks/789"],
            ),
        ],
        answerable_probability=0.7,
    )

    # Act
    synthesizer = GoogleTextSynthesizer()
    response = synthesizer.get_response(
        query_str="What is the meaning of life?",
        text_chunks=[
            "It's 42",
        ],
    )

    # Assert
    assert response.answer == "42"
    assert response.attributed_passages == ["Meaning of life is 42."]
    assert response.answerable_probability == pytest.approx(0.7)

    assert mock_generate_text_answer.call_count == 1
    request = mock_generate_text_answer.call_args.args[0]
    assert request.question.text == "What is the meaning of life?"
    assert request.answer_style == genai.AnswerStyle.ABSTRACTIVE
    passages = request.grounding_source.passages.passages
    assert len(passages) == 1
    passage = passages[0]
    assert passage.text == "It's 42"


@pytest.mark.skipif(not has_google, reason=SKIP_TEST_REASON)
@patch("google.ai.generativelanguage.TextServiceClient.generate_text_answer")
def test_synthesize(mock_generate_text_answer: MagicMock) -> None:
    # Arrange
    mock_generate_text_answer.return_value = genai.GenerateTextAnswerResponse(
        answer=genai.TextCompletion(
            output="42",
            citation_metadata=genai.CitationMetadata(
                citation_sources=[
                    genai.CitationSource(
                        start_index=100,
                        end_index=200,
                        uri="answer.com/meaning_of_life.txt",
                    )
                ]
            ),
        ),
        attributed_passages=[
            genai.AttributedPassage(
                text="Meaning of life is 42",
                passage_ids=["corpora/123/documents/456/chunks/789"],
            ),
            genai.AttributedPassage(
                text="Or maybe not",
                passage_ids=["corpora/123/documents/456/chunks/789"],
            ),
        ],
        answerable_probability=0.9,
    )

    # Act
    synthesizer = GoogleTextSynthesizer()
    response = synthesizer.synthesize(
        query="What is the meaning of life?",
        nodes=[
            NodeWithScore(
                node=TextNode(text="It's 42"),
                score=0.5,
            ),
        ],
        additional_source_nodes=[
            NodeWithScore(
                node=TextNode(text="Additional node"),
                score=0.4,
            ),
        ],
    )

    # Assert
    assert response.response == "42"
    assert len(response.source_nodes) == 4

    first_attributed_source = response.source_nodes[0]
    assert first_attributed_source.node.text == "Meaning of life is 42"
    assert first_attributed_source.score is None

    second_attributed_source = response.source_nodes[1]
    assert second_attributed_source.node.text == "Or maybe not"
    assert second_attributed_source.score is None

    first_input_source = response.source_nodes[2]
    assert first_input_source.node.text == "It's 42"
    assert first_input_source.score == pytest.approx(0.5)

    first_additional_source = response.source_nodes[3]
    assert first_additional_source.node.text == "Additional node"
    assert first_additional_source.score == pytest.approx(0.4)

    assert response.metadata is not None
    assert response.metadata.get("answerable_probability", None) == pytest.approx(0.9)
