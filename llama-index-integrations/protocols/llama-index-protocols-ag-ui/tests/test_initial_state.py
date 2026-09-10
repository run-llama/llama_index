import asyncio

from llama_index.core.llms.mock import MockFunctionCallingLLM
from llama_index.protocols.ag_ui.agent import AGUIChatWorkflow, _copy_initial_state
from llama_index.protocols.ag_ui.router import get_default_workflow_factory


def test_workflow_copies_initial_state_from_caller():
    source_state = {"items": [], "user": {"roles": []}}
    workflow = AGUIChatWorkflow(
        llm=MockFunctionCallingLLM(),
        initial_state=source_state,
    )

    workflow.initial_state["items"].append("alpha")
    workflow.initial_state["user"]["roles"].append("admin")

    assert source_state == {"items": [], "user": {"roles": []}}


def test_default_workflow_factory_returns_isolated_initial_state():
    source_state = {"items": [], "user": {"roles": []}}
    factory = get_default_workflow_factory(
        llm=MockFunctionCallingLLM(),
        initial_state=source_state,
    )

    first = asyncio.run(factory())
    second = asyncio.run(factory())
    first.initial_state["items"].append("alpha")
    first.initial_state["user"]["roles"].append("admin")

    assert second.initial_state == {"items": [], "user": {"roles": []}}
    assert source_state == {"items": [], "user": {"roles": []}}


def test_copy_initial_state_copies_nested_values():
    source_state = {"items": [], "user": {"roles": []}}

    first = _copy_initial_state(source_state)
    second = _copy_initial_state(source_state)
    first["items"].append("alpha")
    first["user"]["roles"].append("admin")

    assert second == {"items": [], "user": {"roles": []}}
    assert source_state == {"items": [], "user": {"roles": []}}
