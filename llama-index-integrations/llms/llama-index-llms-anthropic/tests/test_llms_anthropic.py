import os
import httpx
from unittest.mock import MagicMock

import pytest
from pathlib import Path
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.llms import (
    ChatMessage,
    DocumentBlock,
    TextBlock,
    MessageRole,
    ChatResponse,
)
from llama_index.core.tools import FunctionTool
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.anthropic.base import AnthropicChatResponse


def test_text_inference_embedding_class():
    names_of_base_classes = [b.__name__ for b in Anthropic.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


@pytest.mark.skipif(
    os.getenv("ANTHROPIC_PROJECT_ID") is None,
    reason="Project ID not available to test Vertex AI integration",
)
def test_anthropic_through_vertex_ai():
    anthropic_llm = Anthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet@20240620"),
        region=os.getenv("ANTHROPIC_REGION", "europe-west1"),
        project_id=os.getenv("ANTHROPIC_PROJECT_ID"),
    )

    completion_response = anthropic_llm.complete("Give me a recipe for banana bread")

    try:
        assert isinstance(completion_response.text, str)
        print("Assertion passed for completion_response.text")
    except AssertionError:
        print(
            f"Assertion failed for completion_response.text: {completion_response.text}"
        )
        raise


@pytest.mark.skipif(
    os.getenv("ANTHROPIC_AWS_REGION") is None,
    reason="AWS region not available to test Bedrock integration",
)
def test_anthropic_through_bedrock():
    anthropic_llm = Anthropic(
        aws_region=os.getenv("ANTHROPIC_AWS_REGION", "us-east-1"),
        model=os.getenv("ANTHROPIC_MODEL", "anthropic.claude-3-5-sonnet-20240620-v1:0"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    completion_response = anthropic_llm.complete("Give me a recipe for banana bread")
    print("testing completion")
    try:
        assert isinstance(completion_response.text, str)
        print("Assertion passed for completion_response.text")
    except AssertionError:
        print(
            f"Assertion failed for completion_response.text: {completion_response.text}"
        )
        raise

    # Test streaming completion
    stream_resp = anthropic_llm.stream_complete(
        "Answer in 5 sentences or less. Paul Graham is "
    )
    full_response = ""
    for chunk in stream_resp:
        full_response += chunk.delta

    try:
        assert isinstance(full_response, str)
        print("Assertion passed: full_response is a string")
    except AssertionError:
        print(f"Assertion failed: full_response is not a string")
        print(f"Type of full_response: {type(full_response)}")
        print(f"Content of full_response: {full_response}")
        raise

    messages = [
        ChatMessage(
            role="system", content="You are a pirate with a colorful personality"
        ),
        ChatMessage(role="user", content="Tell me a story"),
    ]

    chat_response = anthropic_llm.chat(messages)
    print("testing chat")
    try:
        assert isinstance(chat_response.message.content, str)
        print("Assertion passed for chat_response")
    except AssertionError:
        print(f"Assertion failed for chat_response: {chat_response}")
        raise

    # Test streaming chat
    stream_chat_resp = anthropic_llm.stream_chat(messages)
    print("testing stream chat")
    full_response = ""
    for chunk in stream_chat_resp:
        full_response += chunk.delta

    try:
        assert isinstance(full_response, str)
        print("Assertion passed: full_response is a string")
    except AssertionError:
        print(f"Assertion failed: full_response is not a string")
        print(f"Type of full_response: {type(full_response)}")
        print(f"Content of full_response: {full_response}")
        raise


@pytest.mark.skipif(
    os.getenv("ANTHROPIC_AWS_REGION") is None,
    reason="AWS region not available to test Bedrock integration",
)
@pytest.mark.asyncio
async def test_anthropic_through_bedrock_async():
    # Note: this assumes you have AWS credentials configured.
    anthropic_llm = Anthropic(
        aws_region=os.getenv("ANTHROPIC_AWS_REGION", "us-east-1"),
        model=os.getenv("ANTHROPIC_MODEL", "anthropic.claude-3-5-sonnet-20240620-v1:0"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    # Test standard async completion
    standard_resp = await anthropic_llm.acomplete(
        "Answer in two sentences or less. Paul Graham is "
    )
    try:
        assert isinstance(standard_resp.text, str)
    except AssertionError:
        print(f"Assertion failed for standard_resp.text: {standard_resp.text}")
        raise

    # Test async streaming
    stream_resp = await anthropic_llm.astream_complete(
        "Answer in 5 sentences or less. Paul Graham is "
    )
    full_response = ""
    async for chunk in stream_resp:
        full_response += chunk.delta

    try:
        assert isinstance(full_response, str)
    except AssertionError:
        print(f"Assertion failed: full_response is not a string")
        print(f"Content of full_response: {full_response}")
        raise
    # Test async chat
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant"),
        ChatMessage(role="user", content="Tell me a short story about AI"),
    ]

    chat_resp = await anthropic_llm.achat(messages)
    try:
        assert isinstance(chat_resp.message.content, str)
    except AssertionError:
        print(f"Assertion failed for chat_resp: {chat_resp}")
        raise

    # Test async streaming chat
    stream_chat_resp = await anthropic_llm.astream_chat(messages)
    full_response = ""
    async for chunk in stream_chat_resp:
        full_response += chunk.delta

    try:
        assert isinstance(full_response, str)
    except AssertionError:
        print(f"Assertion failed: full_response is not a string")
        print(f"Content of full_response: {full_response}")
        raise


def test_anthropic_tokenizer():
    """Test that the Anthropic tokenizer properly implements the Tokenizer protocol."""
    # Create a mock Messages object that returns a predictable token count
    mock_messages = MagicMock()
    mock_messages.count_tokens.return_value.input_tokens = 5

    # Create a mock Beta object that returns our mock messages
    mock_beta = MagicMock()
    mock_beta.messages = mock_messages

    # Create a mock client that returns our mock beta
    mock_client = MagicMock()
    mock_client.beta = mock_beta

    # Create the Anthropic instance with our mock
    anthropic_llm = Anthropic(model="claude-3-5-sonnet-20241022")
    anthropic_llm._client = mock_client

    # Test that tokenizer implements the protocol
    tokenizer = anthropic_llm.tokenizer
    assert hasattr(tokenizer, "encode")

    # Test that encode returns a list of integers
    test_text = "Hello, world!"
    tokens = tokenizer.encode(test_text)
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)
    assert len(tokens) == 5  # Should match our mocked token count

    # Verify the mock was called correctly
    mock_messages.count_tokens.assert_called_once_with(
        messages=[{"role": "user", "content": test_text}],
        model="claude-3-5-sonnet-20241022",
    )


def test__prepare_chat_with_tools_empty():
    llm = Anthropic()
    retval = llm._prepare_chat_with_tools(tools=[])
    assert retval["tools"] == []


@pytest.fixture()
def pdf_url() -> str:
    return "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"


@pytest.mark.skipif(
    os.getenv("ANTHROPIC_API_KEY") is None,
    reason="Anthropic API key not available to test Anthropic integration",
)
def test_tool_required():
    llm = Anthropic(model="claude-3-5-sonnet-latest")

    search_tool = FunctionTool.from_defaults(fn=search)

    # Test with tool_required=True
    response = llm.chat_with_tools(
        user_msg="What is the weather in Paris?",
        tools=[search_tool],
        tool_required=True,
    )
    assert isinstance(response, AnthropicChatResponse)
    assert response.message.additional_kwargs["tool_calls"] is not None
    assert len(response.message.additional_kwargs["tool_calls"]) > 0

    # Test with tool_required=False
    response = llm.chat_with_tools(
        user_msg="Say hello!",
        tools=[search_tool],
        tool_required=False,
    )
    assert isinstance(response, AnthropicChatResponse)
    # Should not use tools for a simple greeting
    assert not response.message.additional_kwargs.get("tool_calls")

    # should not blow up with no tools (regression test)
    response = llm.chat_with_tools(
        user_msg="Say hello!",
        tools=[],
        tool_required=False,
    )
    assert isinstance(response, AnthropicChatResponse)
    assert not response.message.additional_kwargs.get("tool_calls")


@pytest.mark.skipif(
    os.getenv("ANTHROPIC_API_KEY") is None,
    reason="Anthropic API key not available to test Anthropic document uploading ",
)
def test_document_upload(tmp_path: Path, pdf_url: str) -> None:
    llm = Anthropic(model="claude-3-5-sonnet-latest")
    pdf_path = tmp_path / "test.pdf"
    pdf_content = httpx.get(pdf_url).content
    pdf_path.write_bytes(pdf_content)
    msg = ChatMessage(
        role=MessageRole.USER,
        blocks=[
            DocumentBlock(path=pdf_path),
            TextBlock(text="What does the document contain?"),
        ],
    )
    messages = [msg]
    response = llm.chat(messages)
    assert isinstance(response, ChatResponse)


def test_map_tool_choice_to_anthropic():
    """Test that tool_required is correctly mapped to Anthropic's tool_choice parameter."""
    llm = Anthropic()

    # Test with tool_required=True
    tool_choice = llm._map_tool_choice_to_anthropic(
        tool_required=True, allow_parallel_tool_calls=False
    )
    assert tool_choice["type"] == "any"
    assert tool_choice["disable_parallel_tool_use"]

    # Test with tool_required=False
    tool_choice = llm._map_tool_choice_to_anthropic(
        tool_required=False, allow_parallel_tool_calls=False
    )
    assert tool_choice["type"] == "auto"
    assert tool_choice["disable_parallel_tool_use"]

    # Test with allow_parallel_tool_calls=True
    tool_choice = llm._map_tool_choice_to_anthropic(
        tool_required=True, allow_parallel_tool_calls=True
    )
    assert tool_choice["type"] == "any"
    assert not tool_choice["disable_parallel_tool_use"]


def search(query: str) -> str:
    """Search for information about a query."""
    return f"Results for {query}"


search_tool = FunctionTool.from_defaults(
    fn=search, name="search_tool", description="A tool for searching information"
)


def test_prepare_chat_with_tools_tool_required():
    """Test that tool_required is correctly passed to the API request when True."""
    llm = Anthropic()

    # Test with tool_required=True
    result = llm._prepare_chat_with_tools(tools=[search_tool], tool_required=True)

    assert result["tool_choice"]["type"] == "any"
    assert len(result["tools"]) == 1
    assert result["tools"][0]["name"] == "search_tool"


def test_prepare_chat_with_tools_tool_not_required():
    """Test that tool_required is correctly passed to the API request when False."""
    llm = Anthropic()

    # Test with tool_required=False (default)
    result = llm._prepare_chat_with_tools(
        tools=[search_tool],
    )

    assert result["tool_choice"]["type"] == "auto"
    assert len(result["tools"]) == 1
    assert result["tools"][0]["name"] == "search_tool"


def test_prepare_chat_with_no_tools_tool_not_required():
    """Test that tool_required is correctly passed to the API request when False."""
    llm = Anthropic()

    result = llm._prepare_chat_with_tools(tools=[])

    assert "tool_choice" not in result
    assert len(result["tools"]) == 0
