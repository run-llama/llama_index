import asyncio

from ag_ui.core import RunAgentInput, UserMessage

from llama_index.core.llms.mock import MockFunctionCallingLLM
from llama_index.protocols.ag_ui.agent import AGUIChatWorkflow
from llama_index.protocols.ag_ui.router import get_default_workflow_factory


def _llm() -> MockFunctionCallingLLM:
    return MockFunctionCallingLLM(is_chat_model=True)


async def _run_workflow_and_get_state(workflow: AGUIChatWorkflow) -> dict:
    handler = workflow.run(
        input_data=RunAgentInput(
            threadId="thread",
            runId="run",
            state=None,
            messages=[UserMessage(id="user-message", content="hello")],
            tools=[],
            context=[],
            forwardedProps={},
        )
    )
    async for _ in handler.stream_events():
        pass
    await handler
    return await handler.ctx.store.get("state")


def test_default_workflow_factory_copies_initial_state_per_workflow() -> None:
    initial_state = {"items": [], "user": {"roles": []}}

    factory = get_default_workflow_factory(llm=_llm(), initial_state=initial_state)
    first = asyncio.run(factory())
    second = asyncio.run(factory())

    first.initial_state["items"].append("first")
    first.initial_state["user"]["roles"].append("admin")
    first.initial_state["secret"] = "from-first"

    assert second.initial_state == {"items": [], "user": {"roles": []}}
    assert initial_state == {"items": [], "user": {"roles": []}}


def test_default_state_is_copied_for_each_run() -> None:
    workflow = AGUIChatWorkflow(
        llm=_llm(),
        initial_state={"items": [], "user": {"roles": []}},
    )

    first_state = asyncio.run(_run_workflow_and_get_state(workflow))
    first_state["items"].append("first-run")
    first_state["user"]["roles"].append("admin")

    second_state = asyncio.run(_run_workflow_and_get_state(workflow))

    assert second_state == {"items": [], "user": {"roles": []}}
