from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.sglang import SGLang


def test_llm_class():
    names_of_base_classes = [b.__name__ for b in SGLang.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


from unittest.mock import Mock, patch


def test_initialization():
    """Test SGLang initialization."""
    llm = SGLang(
        model="test-model",
        api_url="http://test:8000",
        temperature=0.5,
        max_new_tokens=100,
    )

    assert llm.model == "test-model"
    assert llm.api_url == "http://test:8000"
    assert llm.temperature == 0.5
    assert llm.max_new_tokens == 100


def test_metadata():
    """Test metadata property."""
    llm = SGLang(model="test-model")
    metadata = llm.metadata
    assert metadata.model_name == "test-model"


@patch("llama_index.llms.sglang.base.post_http_request")
@patch("llama_index.llms.sglang.base.get_response")
def test_complete(mock_get_response, mock_post_http_request):
    """Test complete method."""
    mock_response = Mock()
    mock_post_http_request.return_value = mock_response
    mock_get_response.return_value = ["Test response."]

    llm = SGLang(api_url="http://test:8000")
    response = llm.complete("Test prompt")

    assert response.text == "Test response."
    mock_post_http_request.assert_called_once()


@patch("llama_index.llms.sglang.base.post_http_request")
@patch("llama_index.llms.sglang.base.get_response")
def test_chat(mock_get_response, mock_post_http_request):
    """Test chat method."""
    mock_response = Mock()
    mock_post_http_request.return_value = mock_response
    mock_get_response.return_value = ["Chat response."]

    llm = SGLang(api_url="http://test:8000")
    messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
    response = llm.chat(messages)

    assert response.message.content == "Chat response."


@patch("llama_index.llms.sglang.base.post_http_request")
def test_stream_complete(mock_post_http_request):
    """Test stream_complete method."""
    mock_response = Mock()

    chunks = [
        b'data: {"choices": [{"text": "Hello"}]}\n',
        b'data: {"choices": [{"text": " world"}]}\n',
        b"data: [DONE]\n",
    ]
    mock_response.iter_lines.return_value = iter(chunks)
    mock_post_http_request.return_value = mock_response

    llm = SGLang(api_url="http://test:8000")
    gen = llm.stream_complete("Test prompt")

    results = list(gen)
    assert len(results) == 2
    assert results[0].delta == "Hello"
