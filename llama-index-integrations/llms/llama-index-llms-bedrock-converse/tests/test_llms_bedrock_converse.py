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
                yield {"contentBlockDelta": {"delta": {"text": element}}}

        return {"stream": stream_generator()}


class MockClient:
    def __init__(self) -> "MockClient":
        self.exceptions = MockExceptions()

    def converse(self, *args, **kwargs):
        return {"output": {"message": {"content": [{"text": EXP_RESPONSE}]}}}

    def converse_stream(self, *args, **kwargs):
        def stream_generator():
            for element in EXP_STREAM_RESPONSE:
                yield {"contentBlockDelta": {"delta": {"text": element}}}

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
    assert response.additional_kwargs["tool_calls"] == []


def test_stream_chat(bedrock_converse):
    response_stream = bedrock_converse.stream_chat(messages)

    for response in response_stream:
        assert response.message.role == MessageRole.ASSISTANT
        assert response.delta in EXP_STREAM_RESPONSE


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
        assert response.message.role == MessageRole.ASSISTANT
        assert response.delta in EXP_STREAM_RESPONSE


@pytest.mark.asyncio
async def test_acomplete(bedrock_converse):
    response = await bedrock_converse.acomplete(prompt)

    assert isinstance(response, CompletionResponse)
    assert response.text == EXP_RESPONSE
    assert response.additional_kwargs["status"] == []
    assert response.additional_kwargs["tool_call_id"] == []
    assert response.additional_kwargs["tool_calls"] == []


@pytest.mark.asyncio
async def test_astream_complete(bedrock_converse):
    response_stream = await bedrock_converse.astream_complete(prompt)

    responses = []
    async for response in response_stream:
        responses.append(response)

    assert len(responses) == len(EXP_STREAM_RESPONSE)
    assert "".join([r.delta for r in responses]) == "".join(EXP_STREAM_RESPONSE)


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
        tools=[search_tool], tool_required=True, tool_choice=custom_tool_choice
    )

    assert "tools" in result
    assert "toolChoice" in result["tools"]
    assert result["tools"]["toolChoice"] == custom_tool_choice


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
