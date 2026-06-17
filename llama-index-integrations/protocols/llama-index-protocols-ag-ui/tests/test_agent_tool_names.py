import pytest

from llama_index.core.llms.mock import MockFunctionCallingLLM
from llama_index.core.tools import FunctionTool
from llama_index.protocols.ag_ui.agent import AGUIChatWorkflow


def _tool(name: str) -> FunctionTool:
    def tool_fn() -> str:
        return name

    return FunctionTool.from_defaults(fn=tool_fn, name=name)


def test_ag_ui_workflow_rejects_frontend_backend_tool_name_collision() -> None:
    frontend_tool = _tool("shared_tool")
    backend_tool = _tool("shared_tool")

    with pytest.raises(ValueError, match="shared_tool"):
        AGUIChatWorkflow(
            llm=MockFunctionCallingLLM(),
            frontend_tools=[frontend_tool],
            backend_tools=[backend_tool],
        )


def test_ag_ui_workflow_rejects_duplicate_static_tool_names_in_same_group() -> None:
    with pytest.raises(ValueError, match="shared_tool"):
        AGUIChatWorkflow(
            llm=MockFunctionCallingLLM(),
            frontend_tools=[_tool("shared_tool"), _tool("shared_tool")],
        )

    with pytest.raises(ValueError, match="shared_tool"):
        AGUIChatWorkflow(
            llm=MockFunctionCallingLLM(),
            backend_tools=[_tool("shared_tool"), _tool("shared_tool")],
        )


def test_ag_ui_workflow_accepts_unique_frontend_and_backend_tool_names() -> None:
    workflow = AGUIChatWorkflow(
        llm=MockFunctionCallingLLM(),
        frontend_tools=[_tool("frontend_tool")],
        backend_tools=[_tool("backend_tool")],
    )

    assert set(workflow.frontend_tools) == {"frontend_tool"}
    assert set(workflow.backend_tools) == {"backend_tool"}
