from unittest.mock import Mock, patch

import pytest

from llama_index.llms.vertex import Vertex


def _make_mock_gemini_response(
    prompt_tokens: int = 50,
    completion_tokens: int = 20,
    total_tokens: int = 70,
    cached_tokens: int = 0,
    text: str = "Hello world",
) -> Mock:
    """Build a mock Vertex Gemini GenerateContentResponse."""
    mock_usage = Mock()
    mock_usage.prompt_token_count = prompt_tokens
    mock_usage.candidates_token_count = completion_tokens
    mock_usage.total_token_count = total_tokens
    mock_usage.cached_content_token_count = cached_tokens

    mock_response = Mock()
    mock_response.usage_metadata = mock_usage
    mock_response.candidates = [Mock(function_calls=[])]
    mock_response.text = text
    mock_response.__dict__ = {"text": text}
    return mock_response


def test_vertex_metadata_function_calling():
    """Test that Vertex LLM metadata correctly identifies Gemini models as function calling models."""
    # This test uses mocks to avoid actual API calls
    from unittest.mock import patch, Mock

    with patch(
        "llama_index.llms.vertex.gemini_utils.create_gemini_client"
    ) as mock_create_client:
        # Test Gemini model
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        llm = Vertex(model="gemini-pro", project="test-project")
        metadata = llm.metadata

        assert metadata.is_function_calling_model is True
        assert metadata.model_name == "gemini-pro"
        assert metadata.is_chat_model is True


def test_vertex_metadata_non_function_calling():
    """Test that Vertex LLM metadata correctly identifies non-Gemini models as non-function calling models."""
    from unittest.mock import patch, Mock

    with patch(
        "vertexai.language_models.ChatModel.from_pretrained"
    ) as mock_from_pretrained:
        mock_chat_client = Mock()
        mock_from_pretrained.return_value = mock_chat_client

        llm = Vertex(model="chat-bison")
        metadata = llm.metadata

        assert metadata.is_function_calling_model is False
        assert metadata.model_name == "chat-bison"
        assert metadata.is_chat_model is True


def test_extract_usage_metadata_with_tokens():
    """_extract_usage_metadata populates prompt/completion/total tokens for Gemini."""
    with patch(
        "llama_index.llms.vertex.gemini_utils.create_gemini_client"
    ) as mock_create_client:
        mock_create_client.return_value = Mock()
        llm = Vertex(model="gemini-2.0-flash", project="test-project")

    mock_response = _make_mock_gemini_response(
        prompt_tokens=100, completion_tokens=30, total_tokens=130
    )
    usage = llm._extract_usage_metadata(mock_response)

    assert usage["prompt_tokens"] == 100
    assert usage["completion_tokens"] == 30
    assert usage["total_tokens"] == 130
    assert "cached_content_token_count" not in usage


def test_extract_usage_metadata_with_cache_hits():
    """_extract_usage_metadata includes cached_content_token_count when non-zero."""
    with patch(
        "llama_index.llms.vertex.gemini_utils.create_gemini_client"
    ) as mock_create_client:
        mock_create_client.return_value = Mock()
        llm = Vertex(model="gemini-2.0-flash", project="test-project")

    mock_response = _make_mock_gemini_response(
        prompt_tokens=200, completion_tokens=40, total_tokens=240, cached_tokens=150
    )
    usage = llm._extract_usage_metadata(mock_response)

    assert usage["prompt_tokens"] == 200
    assert usage["completion_tokens"] == 40
    assert usage["total_tokens"] == 240
    assert usage["cached_content_token_count"] == 150


def test_extract_usage_metadata_non_gemini_model():
    """_extract_usage_metadata returns empty dict for legacy text/chat-bison models."""
    with patch("vertexai.language_models.ChatModel.from_pretrained") as mock_fp:
        mock_fp.return_value = Mock()
        llm = Vertex(model="chat-bison")

    mock_response = Mock()
    usage = llm._extract_usage_metadata(mock_response)

    assert usage == {}


def test_chat_includes_usage_in_additional_kwargs():
    """chat() propagates token counts into ChatMessage.additional_kwargs."""
    with patch(
        "llama_index.llms.vertex.gemini_utils.create_gemini_client"
    ) as mock_create_client:
        mock_chat_session = Mock()
        mock_client = Mock()
        mock_client.start_chat.return_value = mock_chat_session
        mock_create_client.return_value = mock_client

        mock_response = _make_mock_gemini_response(
            prompt_tokens=50, completion_tokens=20, total_tokens=70
        )
        mock_chat_session.send_message.return_value = mock_response

        llm = Vertex(model="gemini-2.0-flash", project="test-project")
        from llama_index.core.llms import ChatMessage

        response = llm.chat([ChatMessage(role="user", content="Hello")])

    assert response.message.additional_kwargs["prompt_tokens"] == 50
    assert response.message.additional_kwargs["completion_tokens"] == 20
    assert response.message.additional_kwargs["total_tokens"] == 70


def test_chat_includes_cache_tokens_in_additional_kwargs():
    """chat() includes cached_content_token_count when context cache is hit."""
    with patch(
        "llama_index.llms.vertex.gemini_utils.create_gemini_client"
    ) as mock_create_client:
        mock_chat_session = Mock()
        mock_client = Mock()
        mock_client.start_chat.return_value = mock_chat_session
        mock_create_client.return_value = mock_client

        mock_response = _make_mock_gemini_response(
            prompt_tokens=300, completion_tokens=50, total_tokens=350, cached_tokens=250
        )
        mock_chat_session.send_message.return_value = mock_response

        llm = Vertex(model="gemini-2.0-flash", project="test-project")
        from llama_index.core.llms import ChatMessage

        response = llm.chat([ChatMessage(role="user", content="Summarize")])

    assert response.message.additional_kwargs["prompt_tokens"] == 300
    assert response.message.additional_kwargs["completion_tokens"] == 50
    assert response.message.additional_kwargs["total_tokens"] == 350
    assert response.message.additional_kwargs["cached_content_token_count"] == 250
