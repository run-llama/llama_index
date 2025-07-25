import os
import pytest
from dotenv import load_dotenv
from llama_index.core.llms import ChatMessage
from llama_index.llms.cloudflare_ai_gateway import CloudflareAIGateway

# Load .env file
load_dotenv()


@pytest.mark.skipif(
    not all(
        [
            os.getenv("OPENAI_API_KEY"),
            os.getenv("ANTHROPIC_API_KEY"),
            os.getenv("CLOUDFLARE_ACCOUNT_ID"),
            os.getenv("CLOUDFLARE_API_KEY"),
            os.getenv("CLOUDFLARE_GATEWAY"),
        ]
    ),
    reason="Missing required environment variables for real test",
)
def test_real_cloudflare_ai_gateway_with_openai_and_claude():
    """Real test using OpenAI and Claude with Cloudflare AI Gateway fallback."""
    from llama_index.llms.openai import OpenAI
    from llama_index.llms.anthropic import Anthropic

    # Create real LLM instances
    openai_llm = OpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    anthropic_llm = Anthropic(
        model="claude-3-5-sonnet-20241022",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    # Create Cloudflare AI Gateway LLM with fallback order: OpenAI first, then Claude
    llm = CloudflareAIGateway(
        llms=[openai_llm, anthropic_llm],  # Try OpenAI first, then Claude
        account_id=os.getenv("CLOUDFLARE_ACCOUNT_ID"),
        gateway=os.getenv("CLOUDFLARE_GATEWAY"),
        api_key=os.getenv("CLOUDFLARE_API_KEY"),
    )

    # Test chat - Cloudflare AI Gateway will try OpenAI first, then Claude if needed
    messages = [ChatMessage(role="user", content="What is 2+2?")]
    response = llm.chat(messages)

    assert response.message.content is not None
    assert len(response.message.content) > 0
    assert response.message.role == "assistant"

    # Test completion - same fallback behavior
    completion_response = llm.complete("Write a short sentence about AI.")

    assert completion_response.text is not None
    assert len(completion_response.text) > 0

    print("OpenAI/Claude fallback test successful!")
    print(f"Chat response: {response.message.content}")
    print(f"Completion response: {completion_response.text}")
    print(
        "Note: Cloudflare AI Gateway automatically tried OpenAI first, then Claude if needed"
    )


@pytest.mark.skipif(
    not all(
        [
            os.getenv("OPENAI_API_KEY"),
            os.getenv("ANTHROPIC_API_KEY"),
            os.getenv("CLOUDFLARE_ACCOUNT_ID"),
            os.getenv("CLOUDFLARE_API_KEY"),
            os.getenv("CLOUDFLARE_GATEWAY"),
        ]
    ),
    reason="Missing required environment variables for real test",
)
def test_cloudflare_ai_gateway_fallback_when_openai_fails():
    """Test Cloudflare AI Gateway fallback when OpenAI fails."""
    from llama_index.llms.openai import OpenAI
    from llama_index.llms.anthropic import Anthropic

    # Create real LLM instances
    openai_llm = OpenAI(
        model="gpt-4o-mini",
        api_key="invalid-openai-key",  # Invalid key to simulate OpenAI failure
    )

    anthropic_llm = Anthropic(
        model="claude-3-5-sonnet-20241022",
        api_key=os.getenv("ANTHROPIC_API_KEY"),  # Valid Claude key
    )

    # Create Cloudflare AI Gateway LLM with fallback order: OpenAI first, then Claude
    llm = CloudflareAIGateway(
        llms=[openai_llm, anthropic_llm],  # Try OpenAI first (will fail), then Claude
        account_id=os.getenv("CLOUDFLARE_ACCOUNT_ID"),
        gateway=os.getenv("CLOUDFLARE_GATEWAY"),
        api_key=os.getenv("CLOUDFLARE_API_KEY"),
    )

    # Test chat - OpenAI should fail, then fallback to Claude
    messages = [ChatMessage(role="user", content="What is 2+2?")]
    response = llm.chat(messages)

    assert response.message.content is not None
    assert len(response.message.content) > 0
    assert response.message.role == "assistant"

    # Test completion - same fallback behavior
    completion_response = llm.complete("Write a short sentence about AI.")

    assert completion_response.text is not None
    assert len(completion_response.text) > 0

    print("Fallback test successful!")
    print(f"Chat response (from Claude): {response.message.content}")
    print(f"Completion response (from Claude): {completion_response.text}")
    print(
        "Note: OpenAI failed with invalid key, but Claude handled the request successfully"
    )


@pytest.mark.skipif(
    not all(
        [
            os.getenv("OPENAI_API_KEY"),
            os.getenv("ANTHROPIC_API_KEY"),
            os.getenv("CLOUDFLARE_ACCOUNT_ID"),
            os.getenv("CLOUDFLARE_API_KEY"),
            os.getenv("CLOUDFLARE_GATEWAY"),
        ]
    ),
    reason="Missing required environment variables for real test",
)
def test_cloudflare_ai_gateway_fallback_when_both_fail():
    """Test Cloudflare AI Gateway when both providers fail."""
    from llama_index.llms.openai import OpenAI
    from llama_index.llms.anthropic import Anthropic

    # Create LLM instances with invalid keys to simulate failures
    openai_llm = OpenAI(
        model="gpt-4o-mini",
        api_key="invalid-openai-key",
    )

    anthropic_llm = Anthropic(
        model="claude-3-5-sonnet-20241022",
        api_key="invalid-anthropic-key",
    )

    # Create Cloudflare AI Gateway LLM
    llm = CloudflareAIGateway(
        llms=[openai_llm, anthropic_llm],
        account_id=os.getenv("CLOUDFLARE_ACCOUNT_ID"),
        gateway=os.getenv("CLOUDFLARE_GATEWAY"),
        api_key=os.getenv("CLOUDFLARE_API_KEY"),
    )

    # Test that both providers fail and an error is raised
    messages = [ChatMessage(role="user", content="What is 2+2?")]

    with pytest.raises(Exception):  # Should raise an error when both providers fail
        llm.chat(messages)

    print("Both providers failed as expected - error handling works correctly")


@pytest.mark.skipif(
    not all(
        [
            os.getenv("CLOUDFLARE_ACCOUNT_ID"),
            os.getenv("CLOUDFLARE_API_KEY"),
            os.getenv("CLOUDFLARE_GATEWAY"),
        ]
    ),
    reason="Missing required Cloudflare environment variables",
)
def test_cloudflare_ai_gateway_connection():
    """Test basic Cloudflare AI Gateway connection."""
    import httpx

    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    api_key = os.getenv("CLOUDFLARE_API_KEY")
    gateway = os.getenv("CLOUDFLARE_GATEWAY")

    print("Testing connection to Cloudflare AI Gateway:")
    print(f"Account ID: {account_id}")
    print(f"Gateway: {gateway}")
    print(f"API Key: {api_key[:10]}..." if api_key else "None")

    # Test basic connection
    url = f"https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway}"
    headers = {
        "Content-Type": "application/json",
        "cf-aig-authorization": f"Bearer {api_key}",
    }

    # Simple test request
    test_body = [
        {
            "endpoint": "chat/completions",
            "headers": {"Content-Type": "application/json"},
            "provider": "openai",
            "query": {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10,
            },
        }
    ]

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=test_body, headers=headers)
            print(f"Response status: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")

            if response.status_code == 200:
                print("[PASS] Cloudflare AI Gateway connection successful!")
                result = response.json()
                print(f"Response: {result}")
            elif response.status_code == 401:
                print(
                    "[FAIL] Authentication failed - check your API key and permissions"
                )
                print(f"Response: {response.text}")
            elif response.status_code == 404:
                print(
                    "[FAIL] Gateway not found - check your account ID and gateway name"
                )
                print(f"Response: {response.text}")
            else:
                print(f"[FAIL] Unexpected status code: {response.status_code}")
                print(f"Response: {response.text}")

    except Exception as e:
        print(f"[FAIL] Connection error: {e}")
        raise


@pytest.mark.skipif(
    not all(
        [
            os.getenv("OPENAI_API_KEY"),
            os.getenv("CLOUDFLARE_ACCOUNT_ID"),
            os.getenv("CLOUDFLARE_API_KEY"),
            os.getenv("CLOUDFLARE_GATEWAY"),
        ]
    ),
    reason="Missing required environment variables for comprehensive test",
)
def test_cloudflare_ai_gateway_comprehensive_methods():
    """Comprehensive test of all Cloudflare AI Gateway methods with single OpenAI LLM."""
    from llama_index.llms.openai import OpenAI

    # Create single OpenAI LLM instance
    openai_llm = OpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Create Cloudflare AI Gateway LLM with single OpenAI LLM
    llm = CloudflareAIGateway(
        llms=[openai_llm],  # Single OpenAI LLM
        account_id=os.getenv("CLOUDFLARE_ACCOUNT_ID"),
        gateway=os.getenv("CLOUDFLARE_GATEWAY"),
        api_key=os.getenv("CLOUDFLARE_API_KEY"),
    )

    # Store test results
    test_results = []

    # Test 1: Basic chat method
    print("Testing chat method...")
    messages = [ChatMessage(role="user", content="What is 2+2?")]
    chat_response = llm.chat(messages)

    assert chat_response.message.content is not None
    assert len(chat_response.message.content) > 0
    assert chat_response.message.role == "assistant"
    test_results.append(
        ("Basic Chat", "PASS", chat_response.message.content[:50] + "...")
    )

    # Test 2: Basic completion method
    print("Testing completion method...")
    completion_response = llm.complete("Write a short sentence about AI.")

    assert completion_response.text is not None
    assert len(completion_response.text) > 0
    test_results.append(
        ("Basic Completion", "PASS", completion_response.text[:50] + "...")
    )

    # Test 3: Stream chat method
    print("Testing stream chat method...")
    stream_chat_response = llm.stream_chat(messages)
    stream_chat_content = ""
    for chunk in stream_chat_response:
        if hasattr(chunk, "delta") and chunk.delta:
            if hasattr(chunk.delta, "content") and chunk.delta.content:
                stream_chat_content += chunk.delta.content
            elif isinstance(chunk.delta, str):
                stream_chat_content += chunk.delta
        elif hasattr(chunk, "content") and chunk.content:
            stream_chat_content += chunk.content

    assert len(stream_chat_content) > 0
    test_results.append(("Stream Chat", "PASS", stream_chat_content[:50] + "..."))

    # Test 4: Stream completion method
    print("Testing stream completion method...")
    stream_completion_response = llm.stream_complete(
        "Write a short sentence about technology."
    )
    stream_completion_content = ""
    for chunk in stream_completion_response:
        if hasattr(chunk, "delta") and chunk.delta:
            if isinstance(chunk.delta, str):
                stream_completion_content += chunk.delta
            elif hasattr(chunk.delta, "content") and chunk.delta.content:
                stream_completion_content += chunk.delta.content
        elif hasattr(chunk, "content") and chunk.content:
            stream_completion_content += chunk.content
        elif isinstance(chunk, str):
            stream_completion_content += chunk

    assert len(stream_completion_content) > 0
    test_results.append(
        ("Stream Completion", "PASS", stream_completion_content[:50] + "...")
    )

    # Test 5: Metadata property
    print("Testing metadata property...")
    metadata = llm.metadata
    assert metadata is not None
    test_results.append(("Metadata", "PASS", str(metadata.model_name)))

    # Test 6: Class name
    print("Testing class name...")
    class_name = llm.class_name()
    assert class_name == "CloudflareAIGateway"
    test_results.append(("Class Name", "PASS", class_name))

    # Test 7: Chat with system message
    print("Testing chat with system message...")
    system_messages = [
        ChatMessage(
            role="system",
            content="You are a helpful assistant that always responds with 'Hello from AI Gateway!'",
        ),
        ChatMessage(role="user", content="What should you say?"),
    ]
    system_chat_response = llm.chat(system_messages)

    assert system_chat_response.message.content is not None
    assert len(system_chat_response.message.content) > 0
    test_results.append(
        ("System Chat", "PASS", system_chat_response.message.content[:50] + "...")
    )

    # Test 8: Completion with temperature parameter
    print("Testing completion with temperature parameter...")
    temp_completion_response = llm.complete("Write a creative story.", temperature=0.8)

    assert temp_completion_response.text is not None
    assert len(temp_completion_response.text) > 0
    test_results.append(
        ("Temperature Completion", "PASS", temp_completion_response.text[:50] + "...")
    )

    # Test 9: Chat with max_tokens parameter
    print("Testing chat with max_tokens parameter...")
    max_tokens_messages = [
        ChatMessage(role="user", content="Explain quantum computing in detail.")
    ]
    max_tokens_chat_response = llm.chat(max_tokens_messages, max_tokens=50)

    assert max_tokens_chat_response.message.content is not None
    assert len(max_tokens_chat_response.message.content) > 0
    test_results.append(
        (
            "Max Tokens Chat",
            "PASS",
            max_tokens_chat_response.message.content[:50] + "...",
        )
    )

    # Print results table
    print("\n" + "=" * 80)
    print("CLOUDFLARE AI GATEWAY COMPREHENSIVE TEST RESULTS")
    print("=" * 80)
    print(f"{'Test Method':<25} {'Status':<10} {'Sample Response'}")
    print("-" * 80)

    for test_name, status, sample in test_results:
        print(f"{test_name:<25} {status:<10} {sample}")

    print("-" * 80)
    print(f"Total Tests: {len(test_results)} | All Tests: PASS")
    print("=" * 80)
    print("🎉 Cloudflare AI Gateway is working correctly with all methods!")
