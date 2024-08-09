# ./tests/test_github_llm.py

import pytest
from unittest.mock import patch, MagicMock
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
)
from llama_index.llms.githubllm import GithubLLM


@pytest.fixture()
def github_llm():
    return GithubLLM(model="gpt-4o", system_prompt="You are a helpful assistant.")


@pytest.fixture()
def mock_response():
    mock = MagicMock()
    mock.json.return_value = {
        "choices": [{"message": {"content": "This is a test response."}}]
    }
    return mock


@patch.dict(
    "os.environ", {"GITHUB_TOKEN": "fake_token", "AZURE_API_KEY": "fake_azure_key"}
)
@patch("requests.post")
def test_complete(mock_post, github_llm, mock_response):
    mock_post.return_value = mock_response

    response = github_llm.complete("What is the capital of France?")

    assert isinstance(response, CompletionResponse)
    assert response.text == "This is a test response."
    mock_post.assert_called_once()


@patch.dict(
    "os.environ", {"GITHUB_TOKEN": "fake_token", "AZURE_API_KEY": "fake_azure_key"}
)
@patch("requests.post")
def test_chat(mock_post, github_llm, mock_response):
    mock_post.return_value = mock_response

    messages = [ChatMessage(role="user", content="Tell me about Python.")]
    response = github_llm.chat(messages)

    assert isinstance(response, ChatResponse)
    assert response.message.content == "This is a test response."
    assert response.message.role == "assistant"
    mock_post.assert_called_once()


@patch.dict(
    "os.environ", {"GITHUB_TOKEN": "fake_token", "AZURE_API_KEY": "fake_azure_key"}
)
@patch("requests.post")
def test_stream_complete(mock_post, github_llm):
    mock_response = MagicMock()
    mock_response.iter_lines.return_value = [b"chunk1", b"chunk2"]
    mock_post.return_value = mock_response

    generator = github_llm.stream_complete("What is the capital of France?")
    responses = list(generator)

    assert len(responses) == 2
    assert all(isinstance(r, CompletionResponse) for r in responses)
    assert [r.text for r in responses] == ["chunk1", "chunk2"]
    mock_post.assert_called_once()


@patch.dict(
    "os.environ", {"GITHUB_TOKEN": "fake_token", "AZURE_API_KEY": "fake_azure_key"}
)
@patch("requests.post")
def test_stream_chat(mock_post, github_llm):
    mock_response = MagicMock()
    mock_response.iter_lines.return_value = [b"chunk1", b"chunk2"]
    mock_post.return_value = mock_response

    messages = [ChatMessage(role="user", content="Tell me about Python.")]
    generator = github_llm.stream_chat(messages)
    responses = list(generator)

    assert len(responses) == 2
    assert all(isinstance(r, ChatResponse) for r in responses)
    assert [r.message.content for r in responses] == ["chunk1", "chunk2"]
    assert all(r.message.role == "assistant" for r in responses)
    mock_post.assert_called_once()


def test_metadata(github_llm):
    metadata = github_llm.metadata

    assert metadata.model_name == "gpt-4o"
    assert metadata.num_output == 256  # Default value
    assert metadata.context_window == 4096  # Default value


def test_unsupported_model():
    with pytest.raises(ValueError):
        GithubLLM(model="unsupported-model")


@patch.dict("os.environ", {"GITHUB_TOKEN": "fake_token"})
@patch("requests.post")
def test_azure_fallback(mock_post, github_llm, mock_response):
    # Simulate GitHub API failure
    mock_post.side_effect = [Exception("GitHub API Error"), mock_response]

    response = github_llm.complete("What is the capital of France?")

    assert isinstance(response, CompletionResponse)
    assert response.text == "This is a test response."
    assert mock_post.call_count == 2  # Called twice: once for GitHub, once for Azure


@patch.dict("os.environ", {})
def test_missing_env_variables(github_llm):
    with pytest.raises(ValueError):
        github_llm.complete("What is the capital of France?")


def test_class_name(github_llm):
    assert github_llm.class_name() == "GithubLLM"
