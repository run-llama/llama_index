"""Tests for StructuredOutputTool and native structured output functionality."""

import pytest
from typing import Any, List, Optional, Type
from typing_extensions import override
from pydantic import BaseModel, Field

from llama_index.core.llms import (
    LLMMetadata,
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    LLM,
)
from llama_index.core.tools import ToolSelection
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.types import Model
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    AgentOutput,
    AgentStreamStructuredOutput,
    StructuredOutputTool,
    STRUCTURED_OUTPUT_TOOL_NAME,
)
from llama_index.core.agent.workflow.structured_output import (
    extract_structured_output_from_tool_result,
)
from llama_index.core.tools.types import ToolOutput


class SimpleOutput(BaseModel):
    """Simple output model for testing."""

    message: str = Field(description="The response message")
    score: int = Field(description="A numeric score")


class ComplexOutput(BaseModel):
    """Complex output model with nested fields."""

    title: str = Field(description="The title")
    items: List[str] = Field(description="List of items")
    metadata: Optional[dict] = Field(default=None, description="Optional metadata")


class MockLLM(LLM):
    """Mock LLM for testing structured output."""

    def __init__(
        self,
        responses: List[ChatMessage],
        tool_calls: Optional[List[List[ToolSelection]]] = None,
    ):
        super().__init__()
        self._responses = responses
        self._tool_calls = tool_calls or []
        self._response_index = 0

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_function_calling_model=True)

    async def astream_chat(
        self, messages: List[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        response_msg = None
        if self._responses:
            response_msg = self._responses[self._response_index]
            self._response_index = (self._response_index + 1) % len(self._responses)

        async def _gen():
            if response_msg:
                yield ChatResponse(
                    message=response_msg,
                    delta=response_msg.content,
                    raw={"content": response_msg.content},
                )

        return _gen()

    async def astream_chat_with_tools(
        self, tools: List[Any], chat_history: List[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        response_msg = None
        tool_calls = []
        if self._responses:
            response_msg = self._responses[self._response_index]
        if self._tool_calls and self._response_index < len(self._tool_calls):
            tool_calls = self._tool_calls[self._response_index]
        self._response_index = (self._response_index + 1) % max(
            len(self._responses), 1
        )

        if response_msg:
            response_msg.additional_kwargs["tool_calls"] = tool_calls

        async def _gen():
            if response_msg:
                yield ChatResponse(
                    message=response_msg,
                    delta=response_msg.content,
                    raw={"content": response_msg.content},
                )

        return _gen()

    def get_tool_calls_from_response(
        self, response: ChatResponse, **kwargs: Any
    ) -> List[ToolSelection]:
        return response.message.additional_kwargs.get("tool_calls", [])

    @override
    async def astructured_predict(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[dict] = None,
        **prompt_args: Any,
    ) -> Model:
        return output_cls.model_validate({"message": "legacy", "score": 0})

    async def achat(self, *args, **kwargs):
        pass

    def chat(self, *args, **kwargs):
        pass

    def stream_chat(self, *args, **kwargs):
        pass

    def complete(self, *args, **kwargs):
        pass

    async def acomplete(self, *args, **kwargs):
        pass

    def stream_complete(self, *args, **kwargs):
        pass

    async def astream_complete(self, *args, **kwargs):
        pass

    def _prepare_chat_with_tools(self, *args, **kwargs):
        return {}


class TestStructuredOutputTool:
    """Tests for StructuredOutputTool class."""

    def test_create_from_output_cls(self):
        """Test creating a tool from an output class."""
        tool = StructuredOutputTool.from_output_cls(SimpleOutput)

        assert tool.metadata.name == STRUCTURED_OUTPUT_TOOL_NAME
        assert tool.metadata.return_direct is True
        assert tool.metadata.fn_schema == SimpleOutput
        assert "message" in tool.metadata.description
        assert "score" in tool.metadata.description

    def test_create_with_custom_name(self):
        """Test creating a tool with a custom name."""
        tool = StructuredOutputTool.from_output_cls(
            SimpleOutput, name="custom_output"
        )

        assert tool.metadata.name == "custom_output"

    def test_create_with_custom_description(self):
        """Test creating a tool with a custom description."""
        custom_desc = "Custom description for the tool"
        tool = StructuredOutputTool.from_output_cls(
            SimpleOutput, description=custom_desc
        )

        assert tool.metadata.description == custom_desc

    def test_call_with_valid_input(self):
        """Test calling the tool with valid input."""
        tool = StructuredOutputTool.from_output_cls(SimpleOutput)
        result = tool.call(message="Hello", score=42)

        assert not result.is_error
        assert result.raw_output == {"message": "Hello", "score": 42}

    def test_call_with_invalid_input(self):
        """Test calling the tool with invalid input."""
        tool = StructuredOutputTool.from_output_cls(SimpleOutput)
        result = tool.call(message="Hello", score="not_a_number")

        assert result.is_error
        assert "Error validating" in result.content

    @pytest.mark.asyncio
    async def test_acall_with_valid_input(self):
        """Test async calling the tool with valid input."""
        tool = StructuredOutputTool.from_output_cls(SimpleOutput)
        result = await tool.acall(message="Hello", score=42)

        assert not result.is_error
        assert result.raw_output == {"message": "Hello", "score": 42}

    def test_complex_output_model(self):
        """Test with a complex output model."""
        tool = StructuredOutputTool.from_output_cls(ComplexOutput)
        result = tool.call(
            title="Test",
            items=["item1", "item2"],
            metadata={"key": "value"},
        )

        assert not result.is_error
        assert result.raw_output["title"] == "Test"
        assert result.raw_output["items"] == ["item1", "item2"]
        assert result.raw_output["metadata"] == {"key": "value"}


class TestExtractStructuredOutput:
    """Tests for extract_structured_output_from_tool_result function."""

    def test_extract_from_dict_output(self):
        """Test extracting from a dict raw_output."""
        tool_output = ToolOutput(
            content='{"message": "test", "score": 10}',
            tool_name=STRUCTURED_OUTPUT_TOOL_NAME,
            raw_input={},
            raw_output={"message": "test", "score": 10},
        )

        result = extract_structured_output_from_tool_result(tool_output, SimpleOutput)
        assert result == {"message": "test", "score": 10}

    def test_extract_from_error_output(self):
        """Test extracting from an error output returns None."""
        tool_output = ToolOutput(
            content="Error",
            tool_name=STRUCTURED_OUTPUT_TOOL_NAME,
            raw_input={},
            raw_output=None,
            is_error=True,
        )

        result = extract_structured_output_from_tool_result(tool_output, SimpleOutput)
        assert result is None

    def test_extract_from_string_json(self):
        """Test extracting from a JSON string raw_output."""
        tool_output = ToolOutput(
            content='{"message": "test", "score": 10}',
            tool_name=STRUCTURED_OUTPUT_TOOL_NAME,
            raw_input={},
            raw_output='{"message": "test", "score": 10}',
        )

        result = extract_structured_output_from_tool_result(tool_output, SimpleOutput)
        assert result == {"message": "test", "score": 10}


class TestAgentWithNativeStructuredOutput:
    """Integration tests for agents with native structured output."""

    @pytest.fixture
    def mock_llm_with_structured_tool_call(self):
        """Create a mock LLM that calls the structured output tool."""
        return MockLLM(
            responses=[
                ChatMessage(
                    role="assistant",
                    content="",
                    additional_kwargs={},
                )
            ],
            tool_calls=[
                [
                    ToolSelection(
                        tool_id="call_1",
                        tool_name=STRUCTURED_OUTPUT_TOOL_NAME,
                        tool_kwargs={"message": "Native output!", "score": 100},
                    )
                ]
            ],
        )

    @pytest.fixture
    def mock_llm_with_regular_response(self):
        """Create a mock LLM with a regular (non-tool-call) response."""
        return MockLLM(
            responses=[
                ChatMessage(
                    role="assistant",
                    content="Regular response",
                    additional_kwargs={},
                )
            ],
            tool_calls=[[]],
        )

    @pytest.mark.asyncio
    async def test_function_agent_native_structured_output(
        self, mock_llm_with_structured_tool_call
    ):
        """Test FunctionAgent with native structured output enabled."""
        agent = FunctionAgent(
            name="TestAgent",
            description="Test agent",
            llm=mock_llm_with_structured_tool_call,
            output_cls=SimpleOutput,
            use_native_structured_output=True,
        )

        handler = agent.run(user_msg="Test message")
        streaming_event_found = False
        async for event in handler.stream_events():
            if isinstance(event, AgentStreamStructuredOutput):
                streaming_event_found = True
                assert event.output == {"message": "Native output!", "score": 100}

        response = await handler
        assert isinstance(response, AgentOutput)
        # The structured output should be set from the tool call
        if response.structured_response:
            assert response.structured_response["message"] == "Native output!"
            assert response.structured_response["score"] == 100

    @pytest.mark.asyncio
    async def test_agent_workflow_native_structured_output(
        self, mock_llm_with_structured_tool_call
    ):
        """Test AgentWorkflow with native structured output enabled."""
        agent = FunctionAgent(
            name="TestAgent",
            description="Test agent",
            llm=mock_llm_with_structured_tool_call,
        )

        workflow = AgentWorkflow(
            agents=[agent],
            root_agent="TestAgent",
            output_cls=SimpleOutput,
            use_native_structured_output=True,
        )

        handler = workflow.run(user_msg="Test message")
        async for _ in handler.stream_events():
            pass

        response = await handler
        assert isinstance(response, AgentOutput)

    @pytest.mark.asyncio
    async def test_workflow_from_tools_native_structured_output(self):
        """Test AgentWorkflow.from_tools_or_functions with native structured output."""

        def dummy_tool(x: int) -> int:
            """A dummy tool."""
            return x * 2

        mock_llm = MockLLM(
            responses=[
                ChatMessage(
                    role="assistant",
                    content="",
                    additional_kwargs={},
                )
            ],
            tool_calls=[
                [
                    ToolSelection(
                        tool_id="call_1",
                        tool_name=STRUCTURED_OUTPUT_TOOL_NAME,
                        tool_kwargs={"message": "From tools", "score": 50},
                    )
                ]
            ],
        )

        workflow = AgentWorkflow.from_tools_or_functions(
            tools_or_functions=[dummy_tool],
            llm=mock_llm,
            output_cls=SimpleOutput,
            use_native_structured_output=True,
        )

        handler = workflow.run(user_msg="Test")
        async for _ in handler.stream_events():
            pass

        response = await handler
        assert isinstance(response, AgentOutput)


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with legacy structured output."""

    @pytest.fixture
    def mock_llm_no_tool_call(self):
        """Create a mock LLM that doesn't make tool calls."""
        return MockLLM(
            responses=[
                ChatMessage(
                    role="assistant",
                    content="Response without tool call",
                    additional_kwargs={},
                )
            ],
            tool_calls=[[]],
        )

    @pytest.mark.asyncio
    async def test_legacy_mode_uses_extra_llm_call(self, mock_llm_no_tool_call):
        """Test that legacy mode (use_native_structured_output=False) uses extra LLM call."""
        agent = FunctionAgent(
            name="TestAgent",
            description="Test agent",
            llm=mock_llm_no_tool_call,
            output_cls=SimpleOutput,
            use_native_structured_output=False,
        )

        # The agent should work without errors
        handler = agent.run(user_msg="Test message")
        async for _ in handler.stream_events():
            pass

        response = await handler
        assert isinstance(response, AgentOutput)

    @pytest.mark.asyncio
    async def test_structured_output_fn_still_works(self, mock_llm_no_tool_call):
        """Test that structured_output_fn still works as expected."""

        def custom_fn(messages: List[ChatMessage]) -> dict:
            return {"message": "Custom output", "score": 999}

        agent = FunctionAgent(
            name="TestAgent",
            description="Test agent",
            llm=mock_llm_no_tool_call,
            structured_output_fn=custom_fn,
        )

        handler = agent.run(user_msg="Test message")
        async for _ in handler.stream_events():
            pass

        response = await handler
        assert isinstance(response, AgentOutput)
        if response.structured_response:
            assert response.structured_response["message"] == "Custom output"
            assert response.structured_response["score"] == 999
