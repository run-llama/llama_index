"""
Tools sent in ``RunAgentInput.tools`` must reach the LLM with the JSON Schema
the client sent: types, enums, bounds and nested objects included.
"""

import copy

from ag_ui.core import Tool

from llama_index.protocols.ag_ui.agent import _ag_ui_tool_to_llama_index

PARAMETERS = {
    "type": "object",
    "properties": {
        "state": {
            "type": "string",
            "enum": ["on", "off"],
            "description": "Switch state",
        },
        "level": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Percent",
        },
        "color": {
            "type": "object",
            "properties": {"hue": {"type": "number"}},
            "required": ["hue"],
            "description": "Color",
        },
        "tags": {"type": "array", "items": {"type": "string"}},
        "dry_run": {"type": "boolean"},
    },
    "required": ["state", "level"],
}


def test_dynamic_tool_keeps_the_client_schema() -> None:
    tool = Tool(name="set_level", description="Set a level.", parameters=PARAMETERS)
    llama_tool = _ag_ui_tool_to_llama_index(tool)

    assert llama_tool.metadata.get_parameters_dict() == PARAMETERS
    assert llama_tool.metadata.to_openai_tool()["function"]["parameters"] == PARAMETERS


def test_dynamic_tool_schema_is_not_shared_or_mutated() -> None:
    sent = copy.deepcopy(PARAMETERS)
    tool = Tool(name="set_level", description="Set a level.", parameters=sent)
    other = Tool(
        name="other",
        description="Other.",
        parameters={"type": "object", "properties": {"x": {"type": "number"}}},
    )
    first = _ag_ui_tool_to_llama_index(tool)
    second = _ag_ui_tool_to_llama_index(other)

    first.metadata.get_parameters_dict()["properties"].clear()
    assert first.metadata.get_parameters_dict() == PARAMETERS
    assert second.metadata.get_parameters_dict()["properties"] == {
        "x": {"type": "number"}
    }
    assert sent == PARAMETERS


def test_dynamic_tool_without_parameters_gets_an_empty_object_schema() -> None:
    tool = Tool(name="ping", description="Ping.", parameters={})
    assert _ag_ui_tool_to_llama_index(tool).metadata.get_parameters_dict() == {
        "type": "object",
        "properties": {},
    }


def test_dynamic_tool_accepts_the_arguments_the_llm_sends() -> None:
    tool = Tool(name="set_level", description="Set a level.", parameters=PARAMETERS)
    llama_tool = _ag_ui_tool_to_llama_index(tool)
    # Frontend tools are never executed server-side, but the call path still
    # runs; typed arguments must not be rejected or coerced.
    assert llama_tool(state="on", level=40, color={"hue": 0.5}).content == ""
