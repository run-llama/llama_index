import pytest
from unittest.mock import patch

from llama_index.llms.openai_like.responses import OpenAILikeResponses
from llama_index.core.tools import FunctionTool


@pytest.fixture
def default_openai_like_responses():
    """Create a default OpenAILikeResponses instance with mocked clients."""
    with (
        patch("llama_index.llms.openai.base.SyncOpenAI"),
        patch("llama_index.llms.openai.base.AsyncOpenAI"),
    ):
        return OpenAILikeResponses(
            model="test-model",
            api_key="fake-api-key",
            api_base="https://test-api.com/v1",
            context_window=4000,
            is_chat_model=True,
            is_function_calling_model=True,
        )


def test_init_and_properties(default_openai_like_responses):
    """Test initialization and property access."""
    llm = default_openai_like_responses

    assert llm.model == "test-model"
    assert llm.api_base == "https://test-api.com/v1"
    assert llm.api_key == "fake-api-key"
    assert llm.context_window == 4000
    assert llm.is_chat_model is True
    assert llm.is_function_calling_model is True

    metadata = llm.metadata
    assert metadata.model_name == "test-model"
    assert metadata.context_window == 4000
    assert metadata.is_chat_model is True
    assert metadata.is_function_calling_model is True


def test_class_name(default_openai_like_responses):
    """Test class name method."""
    llm = default_openai_like_responses
    assert llm.class_name() == "openai_like_responses_llm"


def test_get_model_kwargs(default_openai_like_responses):
    """Test model kwargs generation for responses API."""
    llm = default_openai_like_responses
    llm.instructions = "Test instructions"
    llm.user = "test_user"

    kwargs = llm._get_model_kwargs()

    assert kwargs["model"] == "test-model"
    assert kwargs["instructions"] == "Test instructions"
    assert kwargs["user"] == "test_user"
    assert kwargs["truncation"] == "disabled"
    assert kwargs["store"] is False
    assert isinstance(kwargs["tools"], list)


def test_get_model_kwargs_with_tools(default_openai_like_responses):
    """Test model kwargs with additional tools."""
    llm = default_openai_like_responses

    def test_function(query: str) -> str:
        return f"Result for {query}"

    tool = FunctionTool.from_defaults(fn=test_function)
    tool_dict = {"type": "function", "name": "test_function"}

    kwargs = llm._get_model_kwargs(tools=[tool_dict])

    assert len(kwargs["tools"]) >= 1
    assert tool_dict in kwargs["tools"]


def test_responses_specific_fields():
    """Test that responses-specific fields are properly set."""
    with (
        patch("llama_index.llms.openai.base.SyncOpenAI"),
        patch("llama_index.llms.openai.base.AsyncOpenAI"),
    ):
        llm = OpenAILikeResponses(
            model="test-model",
            api_key="fake-key",
            api_base="https://test-api.com/v1",
            max_output_tokens=1000,
            instructions="Test instructions",
            track_previous_responses=True,
            built_in_tools=[{"type": "web_search"}],
            user="test_user",
        )

    assert llm.max_output_tokens == 1000
    assert llm.instructions == "Test instructions"
    assert llm.track_previous_responses is True
    assert (
        llm.store is True
    )  # Should be set to True when track_previous_responses is True
    assert llm.built_in_tools == [{"type": "web_search"}]
    assert llm.user == "test_user"


def test_track_previous_responses_enables_store():
    """Test that track_previous_responses=True automatically sets store=True."""
    with (
        patch("llama_index.llms.openai.base.SyncOpenAI"),
        patch("llama_index.llms.openai.base.AsyncOpenAI"),
    ):
        llm = OpenAILikeResponses(
            model="test-model",
            api_key="fake-key",
            api_base="https://test-api.com/v1",
            track_previous_responses=True,
            store=False,  # This should be overridden
        )

    assert llm.track_previous_responses is True
    assert llm.store is True  # Should be automatically set to True


if __name__ == "__main__":
    # Run a simple test to verify the class works
    print("Running basic functionality test...")

    with (
        patch("llama_index.llms.openai.base.SyncOpenAI"),
        patch("llama_index.llms.openai.base.AsyncOpenAI"),
    ):
        llm = OpenAILikeResponses(
            model="test-model",
            api_key="test-key",
            api_base="https://test-api.com/v1",
            context_window=4000,
            is_chat_model=True,
            instructions="You are a helpful assistant",
        )

    print(f"✓ Class instantiated: {llm.class_name()}")
    print(f"✓ Model: {llm.model}")
    print(f"✓ API Base: {llm.api_base}")
    print(f"✓ Context Window: {llm.context_window}")
    print(f"✓ Instructions: {llm.instructions}")

    # Test model kwargs
    kwargs = llm._get_model_kwargs()
    expected_keys = {"model", "instructions", "truncation", "store", "tools"}
    assert expected_keys.issubset(kwargs.keys()), (
        f"Missing keys: {expected_keys - kwargs.keys()}"
    )
    print(f"✓ Model kwargs generated correctly")

    print("All basic tests passed!")
