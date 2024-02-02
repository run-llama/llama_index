from unittest.mock import MagicMock, patch

import pytest

try:
    import google.ai.generativelanguage as genai

    has_google = True
except ImportError:
    has_google = False

from llama_index.legacy.response_synthesizers.google.generativeai import (
    GoogleTextSynthesizer,
    set_google_config,
)
from llama_index.legacy.schema import NodeWithScore, TextNode

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
@patch("google.ai.generativelanguage.GenerativeServiceClient.generate_answer")
def test_get_response(mock_generate_answer: MagicMock) -> None:
    # Arrange
    mock_generate_answer.return_value = genai.GenerateAnswerResponse(
        answer=genai.Candidate(
            content=genai.Content(parts=[genai.Part(text="42")]),
            grounding_attributions=[
                genai.GroundingAttribution(
                    content=genai.Content(
                        parts=[genai.Part(text="Meaning of life is 42.")]
                    ),
                    source_id=genai.AttributionSourceId(
                        grounding_passage=genai.AttributionSourceId.GroundingPassageId(
                            passage_id="corpora/123/documents/456/chunks/789",
                            part_index=0,
                        )
                    ),
                ),
            ],
            finish_reason=genai.Candidate.FinishReason.STOP,
        ),
        answerable_probability=0.7,
    )

    # Act
    synthesizer = GoogleTextSynthesizer.from_defaults(
        temperature=0.5,
        answer_style=genai.GenerateAnswerRequest.AnswerStyle.ABSTRACTIVE,
        safety_setting=[
            genai.SafetySetting(
                category=genai.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=genai.SafetySetting.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            )
        ],
    )
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

    assert mock_generate_answer.call_count == 1
    request = mock_generate_answer.call_args.args[0]
    assert request.contents[0].parts[0].text == "What is the meaning of life?"

    assert request.answer_style == genai.GenerateAnswerRequest.AnswerStyle.ABSTRACTIVE

    assert len(request.safety_settings) == 1
    assert (
        request.safety_settings[0].category
        == genai.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT
    )
    assert (
        request.safety_settings[0].threshold
        == genai.SafetySetting.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
    )

    assert request.temperature == 0.5

    passages = request.inline_passages.passages
    assert len(passages) == 1
    passage = passages[0]
    assert passage.content.parts[0].text == "It's 42"


@pytest.mark.skipif(not has_google, reason=SKIP_TEST_REASON)
@patch("google.ai.generativelanguage.GenerativeServiceClient.generate_answer")
def test_synthesize(mock_generate_answer: MagicMock) -> None:
    # Arrange
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
    synthesizer = GoogleTextSynthesizer.from_defaults()
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


@pytest.mark.skipif(not has_google, reason=SKIP_TEST_REASON)
@patch("google.ai.generativelanguage.GenerativeServiceClient.generate_answer")
def test_synthesize_with_max_token_blocking(mock_generate_answer: MagicMock) -> None:
    # Arrange
    mock_generate_answer.return_value = genai.GenerateAnswerResponse(
        answer=genai.Candidate(
            content=genai.Content(parts=[]),
            grounding_attributions=[],
            finish_reason=genai.Candidate.FinishReason.MAX_TOKENS,
        ),
    )

    # Act
    synthesizer = GoogleTextSynthesizer.from_defaults()
    with pytest.raises(Exception) as e:
        synthesizer.synthesize(
            query="What is the meaning of life?",
            nodes=[
                NodeWithScore(
                    node=TextNode(text="It's 42"),
                    score=0.5,
                ),
            ],
        )

    # Assert
    assert "Maximum token" in str(e.value)


@pytest.mark.skipif(not has_google, reason=SKIP_TEST_REASON)
@patch("google.ai.generativelanguage.GenerativeServiceClient.generate_answer")
def test_synthesize_with_safety_blocking(mock_generate_answer: MagicMock) -> None:
    # Arrange
    mock_generate_answer.return_value = genai.GenerateAnswerResponse(
        answer=genai.Candidate(
            content=genai.Content(parts=[]),
            grounding_attributions=[],
            finish_reason=genai.Candidate.FinishReason.SAFETY,
        ),
    )

    # Act
    synthesizer = GoogleTextSynthesizer.from_defaults()
    with pytest.raises(Exception) as e:
        synthesizer.synthesize(
            query="What is the meaning of life?",
            nodes=[
                NodeWithScore(
                    node=TextNode(text="It's 42"),
                    score=0.5,
                ),
            ],
        )

    # Assert
    assert "safety" in str(e.value)


@pytest.mark.skipif(not has_google, reason=SKIP_TEST_REASON)
@patch("google.ai.generativelanguage.GenerativeServiceClient.generate_answer")
def test_synthesize_with_recitation_blocking(mock_generate_answer: MagicMock) -> None:
    # Arrange
    mock_generate_answer.return_value = genai.GenerateAnswerResponse(
        answer=genai.Candidate(
            content=genai.Content(parts=[]),
            grounding_attributions=[],
            finish_reason=genai.Candidate.FinishReason.RECITATION,
        ),
    )

    # Act
    synthesizer = GoogleTextSynthesizer.from_defaults()
    with pytest.raises(Exception) as e:
        synthesizer.synthesize(
            query="What is the meaning of life?",
            nodes=[
                NodeWithScore(
                    node=TextNode(text="It's 42"),
                    score=0.5,
                ),
            ],
        )

    # Assert
    assert "recitation" in str(e.value)


@pytest.mark.skipif(not has_google, reason=SKIP_TEST_REASON)
@patch("google.ai.generativelanguage.GenerativeServiceClient.generate_answer")
def test_synthesize_with_unknown_blocking(mock_generate_answer: MagicMock) -> None:
    # Arrange
    mock_generate_answer.return_value = genai.GenerateAnswerResponse(
        answer=genai.Candidate(
            content=genai.Content(parts=[]),
            grounding_attributions=[],
            finish_reason=genai.Candidate.FinishReason.OTHER,
        ),
    )

    # Act
    synthesizer = GoogleTextSynthesizer.from_defaults()
    with pytest.raises(Exception) as e:
        synthesizer.synthesize(
            query="What is the meaning of life?",
            nodes=[
                NodeWithScore(
                    node=TextNode(text="It's 42"),
                    score=0.5,
                ),
            ],
        )

    # Assert
    assert "Unexpected" in str(e.value)
