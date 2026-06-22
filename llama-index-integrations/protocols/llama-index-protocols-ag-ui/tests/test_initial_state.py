import asyncio

from llama_index.core.llms import MockFunctionCallingLLM
from llama_index.protocols.ag_ui.agent import AGUIChatWorkflow
from llama_index.protocols.ag_ui.router import get_default_workflow_factory


def test_agui_chat_workflow_copies_initial_state_from_caller() -> None:
    initial_state = {"filters": {"tenant_ids": []}}

    workflow = AGUIChatWorkflow(
        llm=MockFunctionCallingLLM(),
        initial_state=initial_state,
    )
    workflow.initial_state["filters"]["tenant_ids"].append("alice")

    assert initial_state == {"filters": {"tenant_ids": []}}


def test_default_workflow_factory_copies_initial_state_per_workflow() -> None:
    initial_state = {"filters": {"tenant_ids": []}}
    workflow_factory = get_default_workflow_factory(
        llm=MockFunctionCallingLLM(),
        initial_state=initial_state,
    )

    async def create_workflows() -> tuple[AGUIChatWorkflow, AGUIChatWorkflow]:
        first = await workflow_factory()
        second = await workflow_factory()
        assert isinstance(first, AGUIChatWorkflow)
        assert isinstance(second, AGUIChatWorkflow)
        return first, second

    first, second = asyncio.run(create_workflows())
    first.initial_state["filters"]["tenant_ids"].append("alice")

    assert second.initial_state == {"filters": {"tenant_ids": []}}
    assert initial_state == {"filters": {"tenant_ids": []}}
