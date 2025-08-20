import pytest

from llama_index.voice_agents.gemini_live.utils import (
    tools_to_gemini_tools,
    tool_to_fn,
    tools_to_functions_dict,
)
from llama_index.core.tools import FunctionTool
from typing import List, Dict, Any
from google.genai import types


def get_weather(location: str) -> str:
    """Get the weather."""
    return "The weather at " + location + " is sunny"


def add(i: int, j: int) -> int:
    """Add two numbers."""
    return i + j


@pytest.fixture()
def tools() -> List[FunctionTool]:
    return [
        FunctionTool.from_defaults(
            name="get_weather",
            description="Get the weather.",
            fn=get_weather,
        ),
        FunctionTool.from_defaults(
            name="add",
            description="Add two numbers.",
            fn=add,
        ),
    ]


@pytest.fixture()
def function_declarations(
    tools: List[FunctionTool],
) -> List[Dict[str, List[Dict[str, Any]]]]:
    return [
        {
            "function_declarations": [
                {
                    "name": tool.metadata.get_name(),
                    "description": tool.metadata.description,
                    "parameters": tool.metadata.get_parameters_dict(),
                }
                for tool in tools
            ]
        }
    ]


def test_tools_to_gemini_tools(
    tools: List[FunctionTool],
    function_declarations: List[Dict[str, List[Dict[str, Any]]]],
):
    assert tools_to_gemini_tools(tools) == function_declarations


def test_tool_to_fn(tools: List[FunctionTool]):
    t0 = tool_to_fn(tools[0])
    assert callable(tool_to_fn(tools[0]))
    fr0 = t0({"location": "Frankfurt"}, "fn-001", "get_weather")
    assert isinstance(fr0, types.FunctionResponse)
    assert fr0.response == {"result": "The weather at Frankfurt is sunny"}
    assert fr0.id == "fn-001"
    assert fr0.name == "get_weather"


def test_tools_to_fn_dict(tools: List[FunctionTool]):
    td = tools_to_functions_dict(tools)
    assert len(td) == 2
    assert "get_weather" in td
    assert callable(td["get_weather"])
    assert "add" in td
    assert callable(td["add"])
    assert td["add"]({"i": 2, "j": 3}, "fn-002", "add") == types.FunctionResponse(
        id="fn-002", name="add", response={"result": 5}
    )
