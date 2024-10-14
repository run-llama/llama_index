from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.anthropic import Anthropic
from llama_index.core.llms import ChatMessage
import os
import pytest


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
    # Note: this assumes you have AWS credentials configured.
    anthropic_llm = Anthropic(
        aws_region=os.getenv("ANTHROPIC_AWS_REGION", "us-east-1"),
        model=os.getenv("ANTHROPIC_MODEL", "anthropic.claude-3-5-sonnet-20240620-v1:0"),
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
@pytest.mark.asyncio()
async def test_anthropic_through_bedrock_async():
    # Note: this assumes you have AWS credentials configured.
    anthropic_llm = Anthropic(
        aws_region=os.getenv("ANTHROPIC_AWS_REGION", "us-east-1"),
        model=os.getenv("ANTHROPIC_MODEL", "anthropic.claude-3-5-sonnet-20240620-v1:0"),
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
