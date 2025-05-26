from llama_index.core.agent.react.output_parser import (
    extract_final_response,
    extract_tool_use,
    parse_action_reasoning_step,
)


def test_parse_action_reasoning_step() -> None:
    mock_input_text = """\
Thought: Gotta use a tool.
Action: tool
Action Input: {'pages': ['coffee'] /* comment */, 'load_kwargs': {}, 'query_str': ''}, along those lines.
"""
    assert parse_action_reasoning_step(mock_input_text).action_input == {
        "pages": ["coffee"],
        "load_kwargs": {},
        "query_str": "",
    }


def test_extract_tool_use() -> None:
    mock_input_text = """\
Thought: I need to use a tool to help me answer the question.
Action: add
Action Input: {"a": 1, "b": 1}
"""
    thought, action, action_input = extract_tool_use(mock_input_text)
    assert thought == "I need to use a tool to help me answer the question."
    assert action == "add"
    assert action_input == '{"a": 1, "b": 1}'


def test_extract_tool_use_multiline() -> None:
    mock_input_text = """\
Thought: I need to use a tool to help me answer the question.

Action: add



Action Input: {"a": 1, "b": 1}
"""
    thought, action, action_input = extract_tool_use(mock_input_text)
    assert thought == "I need to use a tool to help me answer the question."
    assert action == "add"
    assert action_input == '{"a": 1, "b": 1}'


def test_extract_tool_use_with_nested_dicts() -> None:
    mock_input_text = """\
Thought: Gotta use a tool.
Action: tool
Action Input: {"a": 1, "b": {}}
"""
    thought, action, action_input = extract_tool_use(mock_input_text)
    assert thought == "Gotta use a tool."
    assert action == "tool"
    assert action_input == '{"a": 1, "b": {}}'


def test_extract_tool_use_() -> None:
    mock_input_text = """\
Thought: I need to use a tool to help me answer the question.
Action: add
Action Input: QueryEngineTool({"a": 1, "b": 1})
"""
    thought, action, action_input = extract_tool_use(mock_input_text)
    assert thought == "I need to use a tool to help me answer the question."
    assert action == "add"
    assert action_input == '{"a": 1, "b": 1}'


def test_extract_tool_use_extra_action_output() -> None:
    mock_input_text = """\
Thought: I need to use a tool to help me answer the question.
Action: add (add two numbers)
Action Input: {"a": 1, "b": 1}
"""
    thought, action, action_input = extract_tool_use(mock_input_text)
    assert thought == "I need to use a tool to help me answer the question."
    assert action == "add"
    assert action_input == '{"a": 1, "b": 1}'


def test_extract_tool_number() -> None:
    mock_input_text = """\
Thought: I need to use a tool to help me answer the question.
Action: add2
Action Input: {"a": 1, "b": 1}
"""
    thought, action, action_input = extract_tool_use(mock_input_text)
    assert thought == "I need to use a tool to help me answer the question."
    assert action == "add2"
    assert action_input == '{"a": 1, "b": 1}'


def test_extract_tool_use_multiline_action_input() -> None:
    mock_input_text = """\
Thought: I need to use a tool to help me answer the question.
Action: add
Action Input: {
    "a": 1,
    "b": 1
}
"""
    thought, action, action_input = extract_tool_use(mock_input_text)
    assert thought == "I need to use a tool to help me answer the question."
    assert action == "add"
    assert (
        action_input
        == """\
{
    "a": 1,
    "b": 1
}"""
    )


def test_extract_tool_use_spurious_newlines() -> None:
    mock_input_text = """\
Thought: I need to use a tool to help me answer the question.

Action: add

Action Input: {"a": 1, "b": 1}
"""
    thought, action, action_input = extract_tool_use(mock_input_text)
    assert thought == "I need to use a tool to help me answer the question."
    assert action == "add"
    assert action_input == '{"a": 1, "b": 1}'


def test_extract_tool_use_with_Chinese_characters() -> None:
    mock_input_text = """\
Thought: I need to use a tool to help me answer the question.

Action: 加法

Action Input: {"a": 1, "b": 1}
"""
    thought, action, action_input = extract_tool_use(mock_input_text)
    assert thought == "I need to use a tool to help me answer the question."
    assert action == "加法"
    assert action_input == '{"a": 1, "b": 1}'


def test_extract_final_response() -> None:
    mock_input_text = """\
Thought: I have enough information to answer the question without using any more tools.
Answer: 2
"""

    expected_thought = (
        "I have enough information to answer the question without using any more tools."
    )
    thought, answer = extract_final_response(mock_input_text)
    assert thought == expected_thought
    assert answer == "2"


def test_extract_final_response_multiline_answer() -> None:
    mock_input_text = """\
Thought: I have enough information to answer the question without using any more tools.
Answer: Here is the answer:

This is the second line.
"""

    expected_thought = (
        "I have enough information to answer the question without using any more tools."
    )
    thought, answer = extract_final_response(mock_input_text)
    assert thought == expected_thought
    assert (
        answer
        == """Here is the answer:

This is the second line."""
    )
