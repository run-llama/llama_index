import pytest
import asyncio
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from .base import Bedrock

# Test configuration - Please replace with your actual
MODEL_ID = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"  # For example: anthropic.claude-v2, amazon.titan-text-express-v1, etc.
REGION = "us-west-2"  # Your AWS region
ACCESS_KEY = "xxx"
SECRET_KEY = "xxx"


@pytest.fixture
def bedrock_client():
    """Create Bedrock client instance"""
    return Bedrock(
        model=MODEL_ID,
        region_name=REGION,
        temperature=0.1,
        max_tokens=10000,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )


@pytest.mark.asyncio
async def test_acomplete(bedrock_client):
    """Test asynchronous completion method acomplete"""
    # Execute asynchronous completion request
    prompt = "Explain in one sentence what Python asynchronous programming is"
    response = await bedrock_client.acomplete(prompt)

    # Verify response
    print(f"\n[acomplete response] {response.text[:100]}...\n")

    # Basic assertions
    assert response is not None
    assert isinstance(response.text, str)
    assert len(response.text) > 0

    # Verify if token count is returned (some models may not return it)
    if response.additional_kwargs:
        print(f"Token statistics: {response.additional_kwargs}")


@pytest.mark.asyncio
async def test_astream_complete(bedrock_client):
    """Test asynchronous streaming completion method astream_complete"""
    # Execute asynchronous streaming completion request
    prompt = "List 5 Python asynchronous libraries and their main features"
    response_gen = await bedrock_client.astream_complete(prompt)

    # Collect and process streaming responses
    chunk_count = 0
    final_text = ""

    print("\n[astream_complete streaming response]")
    async for response in response_gen:
        chunk_count += 1
        if chunk_count <= 3 or final_text == "":  # Only print first 3 chunks and final result
            print(f"Chunk #{chunk_count}: Received {len(response.delta)} characters")
            if chunk_count <= 3:
                print(f"Content snippet: {response.delta[:30]}...")
        final_text = response.text

    print(f"Received a total of {chunk_count} response chunks")
    print(f"Final text (first 100 characters): {final_text[:100]}...\n")

    # Basic assertions
    assert chunk_count > 0
    assert isinstance(final_text, str)
    assert len(final_text) > 0


@pytest.mark.asyncio
async def test_achat(bedrock_client):
    """Test asynchronous chat method achat"""
    # Create chat messages
    messages = [
        ChatMessage(role=MessageRole.USER, content="What is the asyncio library? What are its features?")
    ]

    # Execute asynchronous chat request
    response = await bedrock_client.achat(messages)

    # Verify response
    print(f"\n[achat response] {response.message.content[:100]}...\n")

    # Basic assertions
    assert response is not None
    assert response.message is not None
    assert isinstance(response.message.content, str)
    assert len(response.message.content) > 0
    assert response.message.role == MessageRole.ASSISTANT


@pytest.mark.asyncio
async def test_astream_chat(bedrock_client):
    """Test asynchronous streaming chat method astream_chat"""
    # Create chat messages
    messages = [
        ChatMessage(role=MessageRole.USER, content="Explain the purpose of async/await keywords in Python and provide a simple example")
    ]

    # Execute asynchronous streaming chat request
    response_gen = await bedrock_client.astream_chat(messages)

    # Collect and process streaming responses
    chunk_count = 0
    final_content = ""

    print("\n[astream_chat streaming response]")
    async for response in response_gen:
        chunk_count += 1
        if chunk_count <= 3 or final_content == "":  # Only print first 3 chunks and final result
            new_content = response.delta if hasattr(response, 'delta') else response.message.content
            print(f"Chunk #{chunk_count}: Received {len(new_content)} characters")
            if chunk_count <= 3:
                print(f"Content snippet: {new_content[:30]}...")
        final_content = response.message.content

    print(f"Received a total of {chunk_count} response chunks")
    print(f"Final text (first 100 characters): {final_content[:100]}...\n")

    # Basic assertions
    assert chunk_count > 0
    assert isinstance(final_content, str)
    assert len(final_content) > 0


@pytest.mark.asyncio
async def test_multiple_requests(bedrock_client):
    """Test sequential execution of multiple asynchronous requests"""
    # Create message
    message = ChatMessage(role=MessageRole.USER, content="What's the difference between asynchronous programming and multithreading in Python?")

    # Execute multiple asynchronous requests in parallel
    results = await asyncio.gather(
        bedrock_client.acomplete("Explain coroutines in one sentence"),
        bedrock_client.achat([message])
    )

    # Verify results
    complete_response, chat_response = results

    print("\n[Multiple requests test]")
    print(f"acomplete response: {complete_response.text[:50]}...")
    print(f"achat response: {chat_response.message.content[:50]}...")

    # Basic assertions
    assert complete_response is not None
    assert chat_response is not None
    assert len(complete_response.text) > 0
    assert len(chat_response.message.content) > 0


if __name__ == "__main__":
    pytest.main(["-xvs", "test_bedrock_async.py"])