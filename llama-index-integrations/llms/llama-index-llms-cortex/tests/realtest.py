import os
import pytest
import requests
from llama_index.llms.cortex import Cortex
from llama_index.core.llms import ChatMessage, MessageRole
import dotenv

dotenv.load_dotenv()


@pytest.fixture()
def cortex_llm():
    """Create a Cortex LLM instance using environment variables."""
    # Verify that the required environment variable is set
    assert (
        "SNOWFLAKE_PRIVATE_KEY_FILE" in os.environ
    ), "SNOWFLAKE_PRIVATE_KEY_FILE environment variable must be set"

    # Create the Cortex LLM instance
    return Cortex(
        model="llama3.2-1b",  # Using a smaller model for faster tests
        private_key_file=os.environ.get("SNOWFLAKE_PRIVATE_KEY_FILE")
        # Account and user credentials will be taken from environment variables
    )


def test_cortex_metadata(cortex_llm):
    """Test that the LLM metadata is correctly configured."""
    metadata = cortex_llm.metadata

    assert metadata.model_name == "llama3.2-1b"
    assert metadata.is_chat_model is True
    assert metadata.context_window == 128000
    assert metadata.num_output == 4096


def test_urls(cortex_llm: Cortex):
    base_url = cortex_llm.snowflake_api_endpoint
    print(base_url)
    assert base_url.startswith("https://")


def test_cortex_completion(cortex_llm):
    """Test basic completion functionality."""
    prompt = "Write a short haiku about snowflakes."

    response = cortex_llm.complete(prompt)

    # Basic validation checks
    assert response is not None
    assert hasattr(response, "text")
    assert isinstance(response.text, str)
    assert len(response.text) > 0

    print(f"Completion response: {response.text}")


def test_cortex_chat(cortex_llm):
    """Test basic chat functionality."""
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(
            role=MessageRole.USER, content="Write a short haiku about snowflakes."
        ),
    ]

    response = cortex_llm.chat(messages)

    # Basic validation checks
    assert response is not None
    assert hasattr(response, "message")
    assert hasattr(response.message, "content")
    assert isinstance(response.message.content, str)
    assert len(response.message.content) > 0

    print(f"Chat response: {response.message.content}")


def test_cortex_streaming(cortex_llm):
    """Test streaming completion functionality."""
    prompt = "Count from 1 to 5."

    stream_gen = cortex_llm.stream_complete(prompt)

    # Collect all streaming responses
    responses = list(stream_gen)

    # Basic validation checks
    assert len(responses) > 0
    assert all(hasattr(r, "text") for r in responses)
    assert all(isinstance(r.text, str) for r in responses)
    assert all(hasattr(r, "delta") for r in responses)

    # Print the final response
    final_response = responses[-1].text
    print(f"Streaming final response: {final_response}")


def test_cortex_invalid_model_error():
    """Test that using an invalid model name raises an appropriate error."""
    # Create a Cortex LLM instance with an invalid model name
    invalid_model_llm = Cortex(
        model="non-existent-model-xyz",  # This model doesn't exist
        private_key_file=os.environ.get("SNOWFLAKE_PRIVATE_KEY_FILE"),
    )

    # The error should be raised when we try to use the LLM
    with pytest.raises(requests.exceptions.HTTPError):
        invalid_model_llm.complete("This should fail due to invalid model")
