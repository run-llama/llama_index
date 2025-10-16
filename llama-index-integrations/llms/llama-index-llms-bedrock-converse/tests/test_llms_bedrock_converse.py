import random
import string
import json
import os
from llama_index.core.base.llms.types import ImageBlock, TextBlock
import pytest
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    MessageRole,
    CompletionResponse,
    ImageBlock,
    TextBlock,
    ThinkingBlock,
    CachePoint,
    CacheControl,
    ToolCallBlock,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.workflow import Context
from PIL import Image
import io
import numpy as np

# Expected values
EXP_RESPONSE = "Test"
EXP_STREAM_RESPONSE = ["Test ", "value"]
EXP_MAX_TOKENS = 100
EXP_TEMPERATURE = 0.7
EXP_MODEL = "us.anthropic.claude-sonnet-4-20250514-v1:0"
EXP_APP_INF_PROFILE_ARN = "arn:aws:bedrock:us-east-1:012345678901:application-inference-profile/test-profile-name"
EXP_GUARDRAIL_ID = "IDENTIFIER"
EXP_GUARDRAIL_VERSION = "DRAFT"
EXP_GUARDRAIL_TRACE = "ENABLED"

# Reused chat message and prompt
messages = [ChatMessage(role=MessageRole.USER, content="Test")]
prompt = "Test"

# --- Integration Tests ---
# These tests will call the actual AWS Bedrock API.
# They will be skipped if AWS credentials are not found in the environment.
needs_aws_creds = pytest.mark.skipif(
    os.getenv("AWS_ACCESS_KEY_ID") is None
    or os.getenv("AWS_SECRET_ACCESS_KEY") is None
    or os.getenv("AWS_REGION") is None,
    reason="AWS credentials not found in environment, skipping integration test",
)


@pytest.fixture(scope="module")
def temp_image_bytes():
    """Generate a 100x100 red image directly in memory and return its bytes."""
    width, height = 100, 100
    red_array = np.zeros((height, width, 3), dtype=np.uint8)
    red_array[:, :, 0] = 255

    img = Image.fromarray(red_array)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    return buffer.read()


@pytest.fixture(scope="module")
def bedrock_converse_integration():
    """Create a BedrockConverse instance for integration tests with proper credentials."""
    return BedrockConverse(
        model=EXP_MODEL,
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        max_tokens=EXP_MAX_TOKENS,
        system_prompt_caching=True,
    )


@pytest.fixture(scope="module")
def bedrock_converse_integration_thinking():
    """Create a BedrockConverse instance for integration tests with proper credentials."""
    return BedrockConverse(
        model=os.getenv("BEDROCK_THINKING_MODEL")
        or "anthropic.claude-3-7-sonnet-20250219-v1:0",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        max_tokens=12000,
        temperature=1,
        thinking={"budget_tokens": 10000, "type": "enabled"},
    )


@pytest.fixture(scope="module")
def bedrock_converse_integration_no_system_prompt_caching_param():
    """Create a BedrockConverse instance for integration tests with proper credentials."""
    return BedrockConverse(
        model=EXP_MODEL,
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        max_tokens=EXP_MAX_TOKENS,
    )


class MockExceptions:
    class ThrottlingException(Exception):
        pass


class AsyncMockClient:
    def __init__(self) -> "AsyncMockClient":
        self.exceptions = MockExceptions()

    async def __aenter__(self) -> "AsyncMockClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    async def converse(self, *args, **kwargs):
        return {"output": {"message": {"content": [{"text": EXP_RESPONSE}]}}}

    async def converse_stream(self, *args, **kwargs):
        async def stream_generator():
            for element in EXP_STREAM_RESPONSE:
                yield {
                    "contentBlockDelta": {
                        "delta": {"text": element},
                        "contentBlockIndex": 0,
                    }
                }
            # Add messageStop and metadata events for token usage testing
            yield {"messageStop": {"stopReason": "end_turn"}}
            yield {
                "metadata": {
                    "usage": {"inputTokens": 15, "outputTokens": 26, "totalTokens": 41},
                    "metrics": {"latencyMs": 886},
                }
            }

        return {"stream": stream_generator()}


class MockClient:
    def __init__(self) -> "MockClient":
        self.exceptions = MockExceptions()

    def converse(self, *args, **kwargs):
        return {"output": {"message": {"content": [{"text": EXP_RESPONSE}]}}}

    def converse_stream(self, *args, **kwargs):
        def stream_generator():
            for i, element in enumerate(EXP_STREAM_RESPONSE):
                yield {
                    "contentBlockDelta": {
                        "delta": {"text": element},
                        "contentBlockIndex": 0,
                    }
                }
            # Add messageStop and metadata events for token usage testing
            yield {"messageStop": {"stopReason": "end_turn"}}
            yield {
                "metadata": {
                    "usage": {"inputTokens": 15, "outputTokens": 26, "totalTokens": 41},
                    "metrics": {"latencyMs": 886},
                }
            }

        return {"stream": stream_generator()}


class MockAsyncSession:
    def __init__(self, *args, **kwargs) -> "MockAsyncSession":
        pass

    def client(self, *args, **kwargs):
        return AsyncMockClient()


@pytest.fixture()
def mock_boto3_session(monkeypatch):
    def mock_client(*args, **kwargs):
        return MockClient()

    monkeypatch.setattr("boto3.Session.client", mock_client)


@pytest.fixture()
def mock_aioboto3_session(monkeypatch):
    monkeypatch.setattr("aioboto3.Session", MockAsyncSession)


@pytest.fixture()
def bedrock_converse(mock_boto3_session, mock_aioboto3_session):
    return BedrockConverse(
        model=EXP_MODEL,
        max_tokens=EXP_MAX_TOKENS,
        temperature=EXP_TEMPERATURE,
        guardrail_identifier=EXP_GUARDRAIL_ID,
        guardrail_version=EXP_GUARDRAIL_VERSION,
        trace=EXP_GUARDRAIL_TRACE,
        callback_manager=CallbackManager(),
    )


@pytest.fixture()
def bedrock_converse_with_application_inference_profile(
    mock_boto3_session, mock_aioboto3_session
):
    """
    Create a BedrockConverse client that uses an application inference profile for invoking the LLM.
    See AWS documentation for details about creating and using application inference profiles.
    """
    return BedrockConverse(
        model=EXP_MODEL,
        max_tokens=EXP_MAX_TOKENS,
        temperature=EXP_TEMPERATURE,
        guardrail_identifier=EXP_GUARDRAIL_ID,
        guardrail_version=EXP_GUARDRAIL_VERSION,
        application_inference_profile_arn=EXP_APP_INF_PROFILE_ARN,
        trace=EXP_GUARDRAIL_TRACE,
        callback_manager=CallbackManager(),
    )


def test_init(bedrock_converse):
    assert bedrock_converse.model == EXP_MODEL
    assert bedrock_converse.max_tokens == EXP_MAX_TOKENS
    assert bedrock_converse.temperature == EXP_TEMPERATURE
    assert bedrock_converse._client is not None


def test_init_app_inf_profile(bedrock_converse_with_application_inference_profile):
    client = bedrock_converse_with_application_inference_profile
    assert client.application_inference_profile_arn == EXP_APP_INF_PROFILE_ARN
    assert client.model == EXP_MODEL
    # Application inference profile ARN should be used for the LLM kwargs when provided.
    assert client._model_kwargs["model"] == EXP_APP_INF_PROFILE_ARN


def test_chat(bedrock_converse):
    response = bedrock_converse.chat(messages)

    assert response.message.role == MessageRole.ASSISTANT
    assert response.message.content == EXP_RESPONSE


def test_complete(bedrock_converse):
    response = bedrock_converse.complete(prompt)

    assert isinstance(response, CompletionResponse)
    assert response.text == EXP_RESPONSE
    assert response.additional_kwargs["status"] == []
    assert response.additional_kwargs["tool_call_id"] == []


def test_stream_chat(bedrock_converse):
    response_stream = bedrock_converse.stream_chat(messages)

    responses = list(response_stream)

    # Check that we have content responses plus final metadata response
    assert len(responses) == len(EXP_STREAM_RESPONSE) + 1  # +1 for metadata response

    # Check content responses
    for i, response in enumerate(responses[:-1]):  # All except last
        assert response.message.role == MessageRole.ASSISTANT
        assert response.delta in EXP_STREAM_RESPONSE

    # Check final metadata response with token usage
    final_response = responses[-1]
    assert final_response.message.role == MessageRole.ASSISTANT
    assert final_response.delta == ""  # No delta for metadata response

    # Verify raw contains complete metadata
    assert "metadata" in final_response.raw
    assert "usage" in final_response.raw["metadata"]

    # Verify token counts in additional_kwargs
    assert "prompt_tokens" in final_response.additional_kwargs
    assert final_response.additional_kwargs["prompt_tokens"] == 15
    assert final_response.additional_kwargs["completion_tokens"] == 26
    assert final_response.additional_kwargs["total_tokens"] == 41


@pytest.mark.asyncio
async def test_achat(bedrock_converse):
    response = await bedrock_converse.achat(messages)

    assert isinstance(response, ChatResponse)
    assert response.message.role == MessageRole.ASSISTANT
    assert response.message.content == EXP_RESPONSE


@pytest.mark.asyncio
async def test_astream_chat(bedrock_converse):
    response_stream = await bedrock_converse.astream_chat(messages)

    responses = []
    async for response in response_stream:
        responses.append(response)

    # Check that we have content responses plus final metadata response
    assert len(responses) == len(EXP_STREAM_RESPONSE) + 1  # +1 for metadata response

    # Check content responses
    for i, response in enumerate(responses[:-1]):  # All except last
        assert response.message.role == MessageRole.ASSISTANT
        assert response.delta in EXP_STREAM_RESPONSE

    # Check final metadata response with token usage
    final_response = responses[-1]
    assert final_response.message.role == MessageRole.ASSISTANT
    assert final_response.delta == ""  # No delta for metadata response

    # Verify raw contains complete metadata
    assert "metadata" in final_response.raw
    assert "usage" in final_response.raw["metadata"]

    # Verify token counts in additional_kwargs
    assert "prompt_tokens" in final_response.additional_kwargs
    assert final_response.additional_kwargs["prompt_tokens"] == 15
    assert final_response.additional_kwargs["completion_tokens"] == 26
    assert final_response.additional_kwargs["total_tokens"] == 41


@pytest.mark.asyncio
async def test_acomplete(bedrock_converse):
    response = await bedrock_converse.acomplete(prompt)

    assert isinstance(response, CompletionResponse)
    assert response.text == EXP_RESPONSE
    assert response.additional_kwargs["status"] == []
    assert response.additional_kwargs["tool_call_id"] == []


@pytest.mark.asyncio
async def test_astream_complete(bedrock_converse):
    response_stream = await bedrock_converse.astream_complete(prompt)

    responses = []
    async for response in response_stream:
        responses.append(response)

    # Check that we have content responses plus final metadata response
    assert len(responses) == len(EXP_STREAM_RESPONSE) + 1  # +1 for metadata response

    # Check that content responses match expected
    content_responses = responses[:-1]  # All except last (metadata) response
    assert "".join([r.delta for r in content_responses]) == "".join(EXP_STREAM_RESPONSE)


@needs_aws_creds
def test_bedrock_converse_integration_chat_text_only(bedrock_converse_integration):
    """Test a simple text chat integration with Bedrock Converse."""
    llm = bedrock_converse_integration
    messages = [
        ChatMessage(role=MessageRole.USER, content="Write a short sonnet about clouds.")
    ]
    response = llm.chat(messages)

    assert isinstance(response, ChatResponse)
    assert response.message.role == MessageRole.ASSISTANT
    assert isinstance(response.message.content, str)
    assert len(response.message.content) > 5


@needs_aws_creds
def test_bedrock_converse_integration_chat_multimodal(
    temp_image_bytes, bedrock_converse_integration
):
    """Test multimodal chat (text + image) integration with Bedrock Converse."""
    llm = bedrock_converse_integration
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            blocks=[ImageBlock(image=temp_image_bytes, image_mimetype="image/png")],
        ),
        ChatMessage(
            role=MessageRole.USER,
            blocks=[TextBlock(text="What color do you see in the image above?")],
        ),
    ]

    response = llm.chat(messages)

    assert isinstance(response, ChatResponse)
    assert response.message.role == MessageRole.ASSISTANT
    assert isinstance(response.message.content, str)
    assert "red" in response.message.content.lower()


@needs_aws_creds
@pytest.mark.asyncio
async def test_bedrock_converse_integration_achat_text_only(
    bedrock_converse_integration,
):
    """Test async text chat integration."""
    llm = bedrock_converse_integration
    messages = [
        ChatMessage(role=MessageRole.USER, content="What is the capital of France?")
    ]
    response = await llm.achat(messages)

    assert isinstance(response, ChatResponse)
    assert response.message.role == MessageRole.ASSISTANT
    assert isinstance(response.message.content, str)
    assert "paris" in response.message.content.lower()


@needs_aws_creds
@pytest.mark.asyncio
async def test_bedrock_converse_integration_achat_multimodal(
    temp_image_bytes, bedrock_converse_integration
):
    """Test async multimodal chat integration."""
    llm = bedrock_converse_integration

    # Use the red image data from temp_image_bytes fixture
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            blocks=[ImageBlock(image=temp_image_bytes, image_mimetype="image/png")],
        ),
        ChatMessage(
            role=MessageRole.USER,
            blocks=[TextBlock(text="Describe the image provided above briefly.")],
        ),
    ]

    response = await llm.achat(messages)

    assert isinstance(response, ChatResponse)
    assert response.message.role == MessageRole.ASSISTANT
    assert isinstance(response.message.content, str)
    assert len(response.message.content) > 5


@needs_aws_creds
def test_bedrock_converse_integration_stream_chat(bedrock_converse_integration):
    """Test streaming chat integration with Bedrock Converse."""
    llm = bedrock_converse_integration
    messages = [ChatMessage(role=MessageRole.USER, content="Count from 1 to 5 slowly.")]

    response_stream = llm.stream_chat(messages)
    chunks = []
    for response in response_stream:
        chunks.append(response.delta)

    assert len(chunks) > 1
    combined = "".join(chunks)
    assert len(combined) > 5


@needs_aws_creds
def test_bedrock_converse_integration_stream_chat_multimodal(
    temp_image_bytes, bedrock_converse_integration
):
    """Test streaming multimodal chat integration with Bedrock Converse."""
    llm = bedrock_converse_integration
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            blocks=[ImageBlock(image=temp_image_bytes, image_mimetype="image/png")],
        ),
        ChatMessage(
            role=MessageRole.USER,
            blocks=[TextBlock(text="Describe this image in a few words.")],
        ),
    ]

    response_stream = llm.stream_chat(messages)
    chunks = []
    for response in response_stream:
        chunks.append(response.delta)

    assert len(chunks) > 1
    combined = "".join(chunks)
    assert len(combined) > 5


@needs_aws_creds
@pytest.mark.asyncio
async def test_bedrock_converse_integration_astream_chat(bedrock_converse_integration):
    """Test async streaming chat integration with Bedrock Converse."""
    llm = bedrock_converse_integration

    messages = [
        ChatMessage(role=MessageRole.USER, content="Name three famous scientists.")
    ]

    response_stream = await llm.astream_chat(messages)
    chunks = []
    async for response in response_stream:
        chunks.append(response.delta)

    assert len(chunks) > 1
    combined = "".join(chunks)
    assert len(combined) > 5


@needs_aws_creds
@pytest.mark.asyncio
async def test_bedrock_converse_integration_astream_chat_multimodal(
    temp_image_bytes, bedrock_converse_integration
):
    """Test async streaming multimodal chat integration with Bedrock Converse."""
    llm = bedrock_converse_integration
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            blocks=[ImageBlock(image=temp_image_bytes, image_mimetype="image/png")],
        ),
        ChatMessage(
            role=MessageRole.USER,
            blocks=[TextBlock(text="What do you see in this image?")],
        ),
    ]

    response_stream = await llm.astream_chat(messages)
    chunks = []
    async for response in response_stream:
        chunks.append(response.delta)

    assert len(chunks) > 1
    combined = "".join(chunks)
    assert len(combined) > 5


def search(query: str) -> str:
    """Search for information about a query."""
    return f"Results for {query}"


search_tool = FunctionTool.from_defaults(
    fn=search, name="search_tool", description="A tool for searching information"
)


def test_prepare_chat_with_tools_tool_required(bedrock_converse):
    """Test that tool_required=True is correctly passed to the API request."""
    result = bedrock_converse._prepare_chat_with_tools(
        tools=[search_tool], tool_required=True
    )

    assert "tools" in result
    assert "toolChoice" in result["tools"]
    assert result["tools"]["toolChoice"] == {"any": {}}


def test_prepare_chat_with_tools_tool_not_required(bedrock_converse):
    """Test that tool_required=False is correctly passed to the API request."""
    result = bedrock_converse._prepare_chat_with_tools(
        tools=[search_tool], tool_required=False
    )

    assert "tools" in result
    assert "toolChoice" in result["tools"]
    assert result["tools"]["toolChoice"] == {"auto": {}}


def test_prepare_chat_with_tools_custom_tool_choice(bedrock_converse):
    """Test that custom tool_choice overrides tool_required."""
    custom_tool_choice = {"specific": {"name": "search_tool"}}
    result = bedrock_converse._prepare_chat_with_tools(
        tools=[search_tool], tool_choice=custom_tool_choice
    )

    assert "tools" in result
    assert "toolChoice" in result["tools"]
    assert result["tools"]["toolChoice"] == custom_tool_choice


def test_prepare_chat_with_tools_cache_enabled(bedrock_converse):
    """Test that custom tool_choice overrides tool_required."""
    custom_tool_choice = {"specific": {"name": "search_tool"}}
    result = bedrock_converse._prepare_chat_with_tools(
        tools=[search_tool], tool_caching=True
    )

    assert "tools" in result
    assert "toolChoice" in result["tools"]


# Integration test for reproducing the empty text field error
def get_temperature(location: str) -> float:
    """
    A tool that returns the temperature of a given location.

    Args:
        location: The location to get the temperature for.

    Returns:
        The temperature of the location in Celsius.

    """
    return 18.0


@needs_aws_creds
@pytest.mark.asyncio
async def test_bedrock_converse_agent_workflow_empty_text_error(
    bedrock_converse_integration,
):
    """
    Test that reproduces the empty text field error when BedrockConverse
    calls tools without outputting any text in AgentWorkflow.

    This test reproduces the issue described in:
    https://github.com/run-llama/llama_index/issues/18449
    """
    get_temperature_tool = FunctionTool.from_defaults(
        name="get_temperature",
        description="A tool that returns the temperature of a given location.",
        fn=get_temperature,
    )
    agent = FunctionAgent(
        name="weather_agent",
        tools=[get_temperature_tool],
        llm=bedrock_converse_integration,
        system_prompt="You are a helpful assistant that helps users with their queries about the weather.",
    )
    workflow = AgentWorkflow(agents=[agent])

    try:
        response = await workflow.run(
            user_msg="Sort the temperatures of the following locations: Paris, London, Lisbon, Madrid, and Rome."
        )
        assert response is not None

    except Exception as e:
        error_msg = str(e)
        if (
            "The text field in the ContentBlock object" in error_msg
            and "is blank" in error_msg
        ):
            pytest.fail(f"Empty text field error occurred: {error_msg}")
        else:
            raise


@needs_aws_creds
def test_bedrock_converse_integration_chat_with_empty_system_prompt(
    bedrock_converse_integration,
):
    """Test chat integration with empty system prompt."""
    llm = bedrock_converse_integration
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=""),
        ChatMessage(role=MessageRole.USER, content="What is 2 + 2?"),
    ]

    response = llm.chat(messages)

    assert isinstance(response, ChatResponse)
    assert response.message.role == MessageRole.ASSISTANT
    assert isinstance(response.message.content, str)
    assert len(response.message.content) > 0
    assert "4" in response.message.content


@needs_aws_creds
def test_bedrock_converse_integration_chat_with_empty_assistant_message(
    bedrock_converse_integration,
):
    """Test chat integration with empty assistant message in conversation history."""
    llm = bedrock_converse_integration
    messages = [
        ChatMessage(role=MessageRole.USER, content="Hello"),
        ChatMessage(role=MessageRole.ASSISTANT, content=""),
        ChatMessage(role=MessageRole.USER, content="Can you count to 3?"),
    ]

    response = llm.chat(messages)

    assert isinstance(response, ChatResponse)
    assert response.message.role == MessageRole.ASSISTANT
    assert isinstance(response.message.content, str)
    assert len(response.message.content) > 0
    assert any(num in response.message.content for num in ["1", "2", "3"])


@needs_aws_creds
def test_bedrock_converse_integration_chat_with_empty_user_message(
    bedrock_converse_integration,
):
    """Test chat integration with empty user message."""
    llm = bedrock_converse_integration
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content=""),
        ChatMessage(role=MessageRole.USER, content="What is 2 + 2?"),
    ]

    response = llm.chat(messages)

    assert isinstance(response, ChatResponse)
    assert response.message.role == MessageRole.ASSISTANT
    assert isinstance(response.message.content, str)
    assert len(response.message.content) > 0


@needs_aws_creds
@pytest.mark.asyncio
async def test_bedrock_converse_integration_astream_chat_with_empty_assistant_message(
    bedrock_converse_integration,
):
    """Test astream_chat integration with empty assistant message."""
    llm = bedrock_converse_integration

    # Create a conversation with various empty and valid content scenarios
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content="Hello"),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            blocks=[
                TextBlock(text=""),
                TextBlock(text="Previous response"),
            ],
        ),
        ChatMessage(role=MessageRole.USER, content="What is 2+2?"),
    ]

    response_stream = await llm.astream_chat(messages)
    chunks = []
    async for response in response_stream:
        chunks.append(response.delta)

    assert len(chunks) > 0
    combined = "".join(chunks)
    assert len(combined) > 0


# Define a tool function that returns no value
def log_activity(activity: str) -> None:
    """
    Log user activity to system log, but returns no value.

    Args:
        activity: The activity description to log

    """
    print(f"[LOG] User activity: {activity}")
    # This function intentionally returns no value


# Define a tool function that returns no value and takes no arguments
def wake_up_user() -> None:
    """
    Sends a notification to the user to wake them up.
    """
    return


@needs_aws_creds
@pytest.mark.asyncio
async def test_bedrock_converse_agent_with_void_tool_and_continued_conversation(
    bedrock_converse_integration,
):
    """
    Test that Agent can call a tool that returns no value and continue Q&A conversation.

    This test case verifies:
    1. Agent can properly call tools that return no value (void functions)
    2. After calling void tools, the Agent can still answer user questions
    3. No errors occur due to tools not returning values

    This test is important for validating BedrockConverse's handling of tool calls without return values
    """
    # Create a logging tool that returns no value
    log_activity_tool = FunctionTool.from_defaults(
        name="log_activity",
        description="Log user activity to system log for tracking and analysis",
        fn=log_activity,
    )

    # Create a tool with return value for comparison
    get_temperature_tool = FunctionTool.from_defaults(
        name="get_temperature",
        description="Get the temperature of a specified location",
        fn=get_temperature,
    )

    # Create agent using both tools
    agent = FunctionAgent(
        name="assistant_with_logging",
        tools=[log_activity_tool, get_temperature_tool, wake_up_user],
        llm=bedrock_converse_integration,
        system_prompt=(
            "You are a helpful assistant that logs important user activities. "
            "Before answering weather-related questions, please log the user's query activity."
        ),
    )
    workflow = AgentWorkflow(agents=[agent])
    ctx = Context(workflow)

    # First conversation: Request weather information
    # Agent should call log_activity (void tool) first, then call get_temperature
    response1 = await workflow.run(
        user_msg="What's the weather like in San Francisco today? What's the temperature?",
        ctx=ctx,
    )

    # Verify first conversation has normal response
    assert response1 is not None
    assert hasattr(response1, "response")
    response1_text = str(response1.response)
    assert len(response1_text) > 0

    # Second conversation: Continue asking other questions
    # Ensure agent can still handle subsequent conversations after calling void tool
    response2 = await workflow.run(
        user_msg="Will the weather be better tomorrow? Any suggestions?", ctx=ctx
    )

    # Verify second conversation also has normal response
    assert response2 is not None
    assert hasattr(response2, "response")
    response2_text = str(response2.response)
    assert len(response2_text) > 0

    # Third conversation: General question not involving tools
    response3 = await workflow.run(user_msg="Thank you for your help!", ctx=ctx)

    # Verify third conversation response
    assert response3 is not None
    assert hasattr(response3, "response")
    response3_text = str(response3.response)
    assert len(response3_text) > 0

    # Verify blank tool calls are handled correctly
    response_4 = await workflow.run(user_msg="Wake me up please!", ctx=ctx)
    assert response_4 is not None
    assert hasattr(response_4, "response")
    assert len(response_4.tool_calls) > 0

    # Verify all history is handled properly and LLM can continue conversation
    response_5 = await workflow.run(user_msg="Thank you, I am awake now.", ctx=ctx)
    assert response_5 is not None
    assert hasattr(response_5, "response")
    assert len(str(response_5)) > 0


@needs_aws_creds
@pytest.mark.asyncio
async def test_bedrock_converse_thinking(bedrock_converse_integration_thinking):
    messages = [
        ChatMessage(
            role="user",
            content="Can you help me solve this equation for x? x^2+7x+12 = 0. Please think before answering",
        )
    ]
    res_chat = bedrock_converse_integration_thinking.chat(messages)
    assert (
        len(
            [
                block
                for block in res_chat.message.blocks
                if isinstance(block, ThinkingBlock)
            ]
        )
        > 0
    )

    res_achat = await bedrock_converse_integration_thinking.achat(messages)
    assert (
        len(
            [
                block
                for block in res_achat.message.blocks
                if isinstance(block, ThinkingBlock)
            ]
        )
        > 0
    )
    res_stream_chat = bedrock_converse_integration_thinking.stream_chat(messages)

    last_resp = None
    for r in res_stream_chat:
        last_resp = r

    assert all(
        len((block.content or "")) > 10
        for block in last_resp.message.blocks
        if isinstance(block, ThinkingBlock)
    )
    assert len(last_resp.message.blocks) > 0
    res_astream_chat = await bedrock_converse_integration_thinking.astream_chat(
        messages
    )

    last_resp = None
    async for r in res_astream_chat:
        last_resp = r

    assert all(
        len((block.content or "")) > 10
        for block in last_resp.message.blocks
        if isinstance(block, ThinkingBlock)
    )
    assert len(last_resp.message.blocks) > 0


@needs_aws_creds
@pytest.mark.asyncio
async def test_bedrock_converse_integration_system_prompt_cache_points(
    bedrock_converse_integration_no_system_prompt_caching_param,
):
    """
    Test system prompt cache point functionality with BedrockConverse integration.

    This test verifies:
    1. Cache point creation on first call (cache write tokens > 0)
    2. Cache point usage on second call (cache read tokens > 0)
    3. Proper token accounting for cached vs non-cached content

    Uses a system prompt with 1026+ tokens to exceed the 1024 token minimum for caching.
    Each test run uses a unique random identifier to ensure fresh cache creation.
    """
    llm = bedrock_converse_integration_no_system_prompt_caching_param

    # Generate a unique random string for this test run to ensure fresh cache
    # Use fixed length to ensure consistent token counting
    random_id = "".join(random.choices(string.ascii_letters + string.digits, k=8))
    # Create a system prompt with enough tokens for caching
    # Approximate token calculation: "You are a recruiting expert for session ABC12345! " ≈ 11-12 tokens
    base_text = f"You are a recruiting expert for session {random_id}! "

    # Calculate repetitions needed to exceed 1024 tokens (using conservative estimate of 10 tokens per repetition)
    target_tokens = 1100  # Target slightly above minimum to ensure we exceed 1024
    estimated_tokens_per_repetition = 10
    repetitions = target_tokens // estimated_tokens_per_repetition

    repeated_text = base_text * repetitions

    # Additional uncached text to test partial caching
    uncached_instructions = (
        "Please focus on providing helpful responses to job seekers."
    )

    # First call - should establish cache
    cache_test_messages_1 = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            blocks=[
                TextBlock(text=repeated_text),
                CachePoint(cache_control=CacheControl(type="default")),
                TextBlock(text=uncached_instructions),
            ],
        ),
        ChatMessage(
            role=MessageRole.USER, content="Do you have data science jobs in Toronto?"
        ),
    ]

    response_1 = await llm.achat(messages=cache_test_messages_1)
    # Verify cache write tokens are present (first call should write to cache)
    additional_kwargs_1 = getattr(response_1, "additional_kwargs", {})
    assert "cache_creation_input_tokens" in additional_kwargs_1, (
        "First call should show cache creation tokens"
    )
    cache_write_tokens_1 = additional_kwargs_1.get("cache_creation_input_tokens", 0)
    assert cache_write_tokens_1 > 0, (
        f"Expected cache write tokens > 0, got {cache_write_tokens_1}"
    )

    # Second call - should read from cache with different user message
    cache_test_messages_2 = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            blocks=[
                TextBlock(text=repeated_text),  # Same cached content
                CachePoint(cache_control=CacheControl(type="default")),
                TextBlock(text=uncached_instructions),  # Same uncached content
            ],
        ),
        ChatMessage(
            role=MessageRole.USER,
            content="What are the environmental impacts of solar energy?",
        ),
    ]

    response_2 = await llm.achat(messages=cache_test_messages_2)

    # Verify cache read tokens are present (second call should read from cache)
    additional_kwargs_2 = getattr(response_2, "additional_kwargs", {})
    assert "cache_read_input_tokens" in additional_kwargs_2, (
        "Second call should show cache read tokens"
    )
    cache_read_tokens_2 = additional_kwargs_2.get("cache_read_input_tokens", 0)
    assert cache_read_tokens_2 > 0, (
        f"Expected cache read tokens > 0, got {cache_read_tokens_2}"
    )

    # Verify cache efficiency - cache read tokens should be close to cache write tokens
    # (since we're using the same cached content)
    cache_efficiency_ratio = cache_read_tokens_2 / cache_write_tokens_1
    assert 0.95 <= cache_efficiency_ratio <= 1.05, (
        f"Cache efficiency seems off. Write: {cache_write_tokens_1}, "
        f"Read: {cache_read_tokens_2}, Ratio: {cache_efficiency_ratio:.2f}"
    )


@needs_aws_creds
@pytest.mark.asyncio
async def test_bedrock_converse_integration_system_prompt_caching_auto_write(
    bedrock_converse_integration,
):
    """
    Test automatic system prompt cache writing when system_prompt_caching=True.

    This test verifies:
    1. Cache write tokens are properly recorded on first call


    Uses the bedrock_converse_integration fixture which has system_prompt_caching=True.
    Each test run uses a unique random identifier to ensure fresh cache creation.
    """
    llm = bedrock_converse_integration

    # Generate a unique random string for this test run to ensure fresh cache
    # Use fixed length to ensure consistent token counting
    random_id = "".join(random.choices(string.ascii_letters + string.digits, k=8))

    # Create a system prompt with enough tokens for automatic caching
    # The system_prompt_caching=True should automatically cache system prompts >= 1024 tokens
    base_text = (
        f"You are an AI assistant specialized in {random_id} analysis and research. "
    )

    # Calculate repetitions needed to exceed 1024 tokens (using conservative estimate of 12 tokens per repetition)
    target_tokens = 1100  # Target slightly above minimum to ensure we exceed 1024
    estimated_tokens_per_repetition = 12
    repetitions = target_tokens // estimated_tokens_per_repetition

    # Create system prompt that will be automatically cached
    large_system_prompt = base_text * repetitions + (
        "Please provide detailed, helpful, and accurate responses. "
        "Focus on delivering high-quality information with proper context and examples."
    )

    # First call - should trigger automatic cache write for system prompt
    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=large_system_prompt,
        ),
        ChatMessage(
            role=MessageRole.USER,
            content="What are the key benefits of renewable energy?",
        ),
    ]

    response = await llm.achat(messages=messages)

    # Verify cache write tokens are present (first call should write to cache automatically)
    additional_kwargs = getattr(response, "additional_kwargs", {})
    assert "cache_creation_input_tokens" in additional_kwargs, (
        "First call should show cache creation tokens when system_prompt_caching=True"
    )
    cache_write_tokens = additional_kwargs.get("cache_creation_input_tokens", 0)
    assert cache_write_tokens > 0, (
        f"Expected cache write tokens > 0 with automatic caching, got {cache_write_tokens}"
    )

    # Verify response is meaningful
    assert len(str(response.message.content)) > 50, "Response should be substantial"


@needs_aws_creds
@pytest.mark.asyncio
async def test_tool_call_input_output(
    bedrock_converse_integration_thinking: BedrockConverse,
) -> None:
    def get_weather(location: str):
        return f"The weather in {location} is rainy with a temperature of 15°C."

    tool = FunctionTool.from_defaults(
        fn=get_weather,
        name="get_weather",
        description="Get the weather of a given location",
    )

    history = [
        ChatMessage(
            role="user",
            content="Hello, can you tell me what is the weather today in London?",
        ),
        ChatMessage(
            role="assistant",
            blocks=[
                ToolCallBlock(
                    tool_name="get_weather",
                    tool_kwargs={"location": "Liverpool"},
                    tool_call_id="1",
                ),
            ],
        ),
        ChatMessage(
            role=MessageRole.TOOL,
            content="The weather in London is 11°C and windy",
            additional_kwargs={"tool_call_id": "1"},
        ),
        ChatMessage(
            role="assistant",
            blocks=[
                TextBlock(
                    text="The weather in London is windy with a temperature of 11°C"
                )
            ],
        ),
    ]

    input_message = ChatMessage(
        role="user",
        content="Ok, and what is the weather in Liverpool?",
    )

    response = bedrock_converse_integration_thinking.chat_with_tools(
        tools=[tool], user_msg=input_message, chat_history=history
    )
    assert (
        len(
            [
                block
                for block in response.message.blocks
                if isinstance(block, ToolCallBlock)
            ]
        )
        > 0
    )
    assert any(
        block.tool_name == "get_weather"
        and (
            block.tool_kwargs == {"location": "Liverpool"}
            or block.tool_kwargs == json.dumps({"location": "Liverpool"})
        )
        for block in response.message.blocks
        if isinstance(block, ToolCallBlock)
    )
    aresponse = await bedrock_converse_integration_thinking.achat_with_tools(
        tools=[tool], user_msg=input_message, chat_history=history
    )
    assert (
        len(
            [
                block
                for block in aresponse.message.blocks
                if isinstance(block, ToolCallBlock)
            ]
        )
        > 0
    )
    assert any(
        block.tool_name == "get_weather"
        and (
            block.tool_kwargs == {"location": "Liverpool"}
            or block.tool_kwargs == json.dumps({"location": "Liverpool"})
        )
        for block in aresponse.message.blocks
        if isinstance(block, ToolCallBlock)
    )
    stream_response = bedrock_converse_integration_thinking.stream_chat_with_tools(
        tools=[tool], user_msg=input_message, chat_history=history
    )
    blocks = []
    for res in stream_response:
        blocks.extend(res.message.blocks)
    assert len([block for block in blocks if isinstance(block, ToolCallBlock)]) > 0
    assert any(
        block.tool_name == "get_weather"
        and (
            block.tool_kwargs == {"location": "Liverpool"}
            or block.tool_kwargs == json.dumps({"location": "Liverpool"})
        )
        for block in blocks
        if isinstance(block, ToolCallBlock)
    )
    astream_response = (
        await bedrock_converse_integration_thinking.astream_chat_with_tools(
            tools=[tool], user_msg=input_message, chat_history=history
        )
    )
    ablocks = []
    async for res in astream_response:
        ablocks.extend(res.message.blocks)
    assert len([block for block in ablocks if isinstance(block, ToolCallBlock)]) > 0
    assert any(
        block.tool_name == "get_weather"
        and (
            block.tool_kwargs == {"location": "Liverpool"}
            or block.tool_kwargs == json.dumps({"location": "Liverpool"})
        )
        for block in ablocks
        if isinstance(block, ToolCallBlock)
    )
