import pytest

from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.llms.mock import MockFunctionCallingLLM
from llama_index.core.workflow import Context
from llama_index.core.workflow.errors import WorkflowRuntimeError


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("multi_agent", [False, True])
async def test_new_run_does_not_report_tools_from_failed_run(
    streaming: bool, multi_agent: bool
) -> None:
    calls = []

    def record(value: str) -> str:
        calls.append(value)
        return f"recorded {value}"

    responses = iter(
        [
            ChatMessage(
                role="assistant",
                content="recording",
                additional_kwargs={
                    "tool_calls": [
                        ToolSelection(
                            tool_id="old-call",
                            tool_name="record",
                            tool_kwargs={"value": "old"},
                        )
                    ]
                },
            ),
            ChatMessage(role="assistant", content="old answer"),
            ChatMessage(role="assistant", content="new answer"),
        ]
    )

    def respond(messages, **kwargs):
        return next(responses)

    agent = FunctionAgent(
        llm=MockFunctionCallingLLM(response_generator=respond),
        tools=[record],
        streaming=streaming,
    )
    workflow = AgentWorkflow(agents=[agent]) if multi_agent else agent
    ctx = Context(workflow)

    with pytest.raises(WorkflowRuntimeError, match="Max iterations of 2 reached"):
        await workflow.run(user_msg="record old", ctx=ctx, max_iterations=2)

    result = await workflow.run(user_msg="new request without tools", ctx=ctx)

    assert result.response.content == "new answer"
    assert calls == ["old"]
    assert result.tool_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("multi_agent", [False, True])
async def test_new_run_keeps_its_own_tool_results_after_model_failure(
    multi_agent: bool,
) -> None:
    calls = []

    def record(value: str) -> str:
        calls.append(value)
        return f"recorded {value}"

    def tool_response(value: str) -> ChatMessage:
        return ChatMessage(
            role="assistant",
            additional_kwargs={
                "tool_calls": [
                    ToolSelection(
                        tool_id=f"{value}-call",
                        tool_name="record",
                        tool_kwargs={"value": value},
                    )
                ]
            },
        )

    responses = iter(
        [
            tool_response("old"),
            RuntimeError("provider unavailable"),
            tool_response("new-1"),
            tool_response("new-2"),
            ChatMessage(role="assistant", content="new answer"),
        ]
    )

    def respond(messages, **kwargs):
        response = next(responses)
        if isinstance(response, RuntimeError):
            raise response
        return response

    agent = FunctionAgent(
        llm=MockFunctionCallingLLM(response_generator=respond),
        tools=[record],
        streaming=False,
    )
    workflow = AgentWorkflow(agents=[agent]) if multi_agent else agent
    ctx = Context(workflow)

    with pytest.raises(RuntimeError, match="provider unavailable"):
        await workflow.run(user_msg="record old", ctx=ctx)

    result = await workflow.run(user_msg="record two new values", ctx=ctx)

    assert result.response.content == "new answer"
    assert calls == ["old", "new-1", "new-2"]
    assert [tool.tool_id for tool in result.tool_calls] == ["new-1-call", "new-2-call"]
