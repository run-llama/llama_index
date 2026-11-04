import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Any, Sequence


from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    SimpleAgentContext,
)
from llama_index.core.agent.workflow.codeact_agent import CodeActAgent
from llama_index.core.agent.workflow.workflow_events import AgentOutput, ToolCallResult
from llama_index.core.base.llms.types import (
    ChatResponse,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
)
from llama_index.core.llms import ChatMessage, LLMMetadata
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.llms.mock import MockFunctionCallingLLM
from llama_index.core.tools import ToolOutput
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer


@pytest.fixture()
def mock_llm():
    # Create a mock that inherits from FunctionCallingLLM
    class MockFunctionCallingLLM(FunctionCallingLLM):
        get_tool_calls_from_response: Any = MagicMock(return_value=[])

        def __init__(self) -> None:
            super().__init__()
            self._responses = []

        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(
                is_function_calling_model=True,
            )

        async def astream_chat(self, *args, **kwargs):
            # Return an async generator that yields each response
            async def gen():
                for response in self._responses:
                    yield response

            return gen()

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

    return MockFunctionCallingLLM()


@pytest.fixture()
def mock_code_execute_fn():
    return lambda code: "Code executed"


@pytest.fixture()
def mock_memory():
    memory = AsyncMock(spec=BaseMemory)
    memory.aput = AsyncMock()
    return memory


@pytest.mark.asyncio
@pytest.mark.parametrize("split_chunks", [False, True])
async def test_code_act_agent_basic_execution(
    split_chunks, mock_llm, mock_code_execute_fn, mock_memory
):
    # Setup mock response
    mock_response = ChatResponse(
        message=ChatMessage(
            role="assistant",
            content="Let me calculate that for you.\n<execute>\nprint('Hello World')\n</execute>",
        ),
        delta="Let me calculate that for you.\n<execute>\nprint('Hello World')\n</execute>",
    )
    mock_llm._responses = [mock_response]  # Set the responses to be yielded
    if split_chunks:
        text = mock_response.message.content
        mock_llm._responses = [
            ChatResponse(
                message=ChatMessage(role="assistant", content=chunk), delta=chunk
            )
            for chunk in [text[:20], text[20:]]
        ]

    # Create agent
    agent = CodeActAgent(
        code_execute_fn=mock_code_execute_fn,
        llm=mock_llm,
    )

    # Create context
    ctx = SimpleAgentContext()

    # Take step
    output = await agent.take_step(
        ctx=ctx,
        llm_input=[ChatMessage(role="user", content="Say hello")],
        tools=[],
        memory=mock_memory,
    )

    # Verify output
    assert isinstance(output, AgentOutput)
    assert output.response.content == mock_response.message.content
    assert (await ctx.store.get("scratchpad"))[0] == output.response
    assert len(output.tool_calls) == 1
    assert output.tool_calls[0].tool_name == "execute"
    assert "print('Hello World')" in output.tool_calls[0].tool_kwargs["code"]


@pytest.mark.asyncio
async def test_code_act_agent_tool_handling(
    mock_llm, mock_code_execute_fn, mock_memory
):
    # Setup mock response
    mock_response = ChatResponse(
        message=ChatMessage(
            role="assistant",
            content="Let me calculate that for you.\n<execute>\nresult = 2 + 2\nprint(result)\n</execute>",
        ),
        delta="Let me calculate that for you.\n<execute>\nresult = 2 + 2\nprint(result)\n</execute>",
    )
    mock_llm._responses = [mock_response]  # Set the responses to be yielded

    # Create agent
    agent = CodeActAgent(
        code_execute_fn=mock_code_execute_fn,
        llm=mock_llm,
    )

    # Create context
    ctx = SimpleAgentContext()

    # Take step
    output = await agent.take_step(
        ctx=ctx,
        llm_input=[ChatMessage(role="user", content="What is 2 + 2?")],
        tools=[],
        memory=mock_memory,
    )

    # Handle tool results
    tool_results = [
        ToolCallResult(
            tool_id=output.tool_calls[0].tool_id,
            tool_name="execute",
            tool_kwargs={"code": "result = 2 + 2\nprint(result)\n"},
            tool_output=ToolOutput(
                content="4", tool_name="execute", raw_input={}, raw_output={}
            ),
            return_direct=False,
        )
    ]
    await agent.handle_tool_call_results(ctx, tool_results, mock_memory)

    # Verify scratchpad was updated
    scratchpad = await ctx.store.get("scratchpad")
    assert len(scratchpad) == 2  # User message and assistant response
    assert "4" in scratchpad[1].content  # Verify the result was added to scratchpad

    # Finalize
    final_output = await agent.finalize(ctx, output, mock_memory)
    assert isinstance(final_output, AgentOutput)
    assert mock_memory.aput_messages.called  # Verify memory was updated


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
async def test_code_act_agent_workflow_integration(streaming):
    """
    Integration test that runs the agent through the full workflow,
    verifying proper integration with the workflows library.
    """
    # Track call count to return different responses
    call_count = [0]

    def multi_response_generator(
        messages: Sequence[ChatMessage], **kwargs
    ) -> ChatMessage:
        call_count[0] += 1
        if call_count[0] == 1:
            # First call: return code to execute
            return ChatMessage(
                role="assistant",
                content="Let me calculate that.\n<execute>\nresult = 2 + 2\nprint(result)\n</execute>",
            )
        else:
            # Second call: return final answer
            return ChatMessage(
                role="assistant",
                content="The answer is 4.",
            )

    mock_llm = MockFunctionCallingLLM(response_generator=multi_response_generator)

    # Create agent with a code_execute_fn that returns a result
    def execute_code(code: str) -> str:
        return "4"

    agent = CodeActAgent(
        code_execute_fn=execute_code,
        llm=mock_llm,
        streaming=streaming,
    )

    # Run the agent through the full workflow
    handler = agent.run(user_msg="What is 2 + 2?")
    result = await handler

    # Verify we got an AgentOutput
    assert isinstance(result, AgentOutput)
    assert result.response.content == "The answer is 4."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("streaming", "split_chunks"), [(True, False), (False, False), (True, True)]
)
@pytest.mark.parametrize("tool_call_format", ["blocks", "additional_kwargs"])
async def test_code_act_agent_handoff_preserves_transcript(
    streaming, split_chunks, tool_call_format, mock_code_execute_fn
):
    tool_kwargs = {"to_agent": "receiver", "reason": "delegate"}
    handoff_message = ChatMessage(role="assistant")
    if tool_call_format == "blocks":
        handoff_message.blocks = [
            ToolCallBlock(
                tool_call_id="handoff-1",
                tool_name="handoff",
                tool_kwargs=tool_kwargs,
            )
        ]
    else:
        handoff_message.additional_kwargs["tool_calls"] = [
            ToolSelection(
                tool_id="handoff-1",
                tool_name="handoff",
                tool_kwargs=tool_kwargs,
            )
        ]

    expected_message = handoff_message
    if split_chunks:
        handoff_message = handoff_message.model_copy(
            update={
                "blocks": [
                    TextBlock(text="handoff"),
                    ThinkingBlock(content="Delegate to the receiver."),
                    *handoff_message.blocks,
                ]
            }
        )
        expected_message = handoff_message.model_copy(
            update={
                "blocks": [
                    TextBlock(text="before handoff"),
                    *handoff_message.blocks[1:],
                ]
            }
        )

    async def handoff_chunks(messages: Sequence[ChatMessage], **kwargs):
        yield ChatMessage(role="assistant", content="before ")
        yield handoff_message

    def receiver_response(messages: Sequence[ChatMessage], **kwargs) -> ChatMessage:
        assistant_message, tool_message = messages[-2:]
        assert assistant_message.role == "assistant"
        assert tool_message.role == "tool"
        assert tool_message.additional_kwargs["tool_call_id"] == "handoff-1"
        # A provider must receive the tool call before its matching tool result.
        assert assistant_message == expected_message
        return ChatMessage(role="assistant", content="Received handoff.")

    code_agent = CodeActAgent(
        name="coder",
        code_execute_fn=mock_code_execute_fn,
        llm=MockFunctionCallingLLM(
            response_generator=handoff_chunks
            if split_chunks
            else lambda messages, **kwargs: handoff_message
        ),
        can_handoff_to=["receiver"],
        streaming=streaming,
    )
    receiver = FunctionAgent(
        name="receiver",
        description="Receives delegated tasks.",
        llm=MockFunctionCallingLLM(response_generator=receiver_response),
        can_handoff_to=[],
        streaming=streaming,
    )
    workflow = AgentWorkflow(agents=[code_agent, receiver], root_agent="coder")
    memory = ChatMemoryBuffer.from_defaults(tokenizer_fn=lambda text: text.split())

    result = await workflow.run(user_msg="Delegate this task.", memory=memory)

    assert result.response.content == "Received handoff."
    assert memory.get_all()[1] == expected_message
    if split_chunks:
        assert handoff_message.content == "handoff"
