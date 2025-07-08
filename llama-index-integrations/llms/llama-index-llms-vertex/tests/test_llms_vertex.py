from llama_index.llms.vertex import Vertex


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
